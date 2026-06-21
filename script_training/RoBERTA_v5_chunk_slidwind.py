from collections import defaultdict
from statistics import stdev

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from transformers import RobertaForSequenceClassification, RobertaTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm.notebook import tqdm
from dlordinal.losses import TriangularLoss
from labeling_function_v7 import SignalExtractor 

# environment specific configuration
# from env import *

# Styling
sns.set_theme('notebook', style='whitegrid')

#===== Device settings =====
# Check for GPU availability and set device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA")
elif torch.backends.mps.is_available():

    device = torch.device("mps")
    print("MPS")
else:
    device = torch.device("cpu")
    print("CPU")

torch.manual_seed(42)
print(f"Using torch {torch.__version__} on {device}")

# ===== Check memory =====
if torch.cuda.is_available():
#    print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB \n")
#    print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB \n")
   print(f"GPU total memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB \n")

import gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Also delete any leftover model available
try:
    del model
    gc.collect()
    torch.cuda.empty_cache()
except NameError:
    pass

# ===== 0. Experiment CONFIG block ====
# For easier change of settings between runs

CONFIG = {
    'experiment_name': 'v5_chunk_slidwind',
    'max_len': 512,
    'batch_size': 4,
    'epochs': 10,
    'freeze': False,
    'learning_rate': 2e-5,
    # 'dropout': 0.4,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1
}

# ===== 1. Masking and fast-masking =====
def mask_payload(payload):
    if not isinstance(payload, str):
        return ""

    # Loop over reg ex dictionary from the labeling functions script to blank out signals
    for signal_name, pattern in SignalExtractor.regex_patterns.items():
        payload = pattern.sub("", payload)
    return payload

# !!!
def fast_mask_payload(html_string):
    if not isinstance(html_string, str):
        split_point = 5000 
    
    # If the document is small, mask the whole thing
    if len(html_string) <= split_point:
        return mask_payload(html_string)
        
    # Split, mask the top, and concatenate with the untouched bottom
    top_part = html_string[:split_point]
    bottom_part = html_string[split_point:]
    
    cleaned_top = mask_payload(top_part)
    
    return cleaned_top + bottom_part

# ===== 2. Load data and pre-process =====
print("Loading data...")
df = pd.read_json('./training/train_w23.jsonl', lines=True)

# !!!! fast-masking
df['payload'] = df['payload'].apply(fast_mask_payload)
print("Data masked successfully.")

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['period_bucket'])

# splitting is done later in the SKF loop for stratification per fold

# ===== 3. Dataset and Tokenizer =====
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

class ChunkedHTMLDataset(Dataset):
    """Previous method: loop over all documents immediately -> tokenize every document fully -> split into chunks -> store chunks as tensors in RAM.

    Lazy loading: loop over docs -> store only raw text strings and an index -> training loop requests batch (4) -> each call tokenizes ONE document on the spot
    -> extracts requested chunk -> returns tensor -> tensor used in forward pass, then thrown awat."""

    def __init__(self, dataframe, tokenizer, max_len=512, stride=256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stride = stride
        self.index = [] # (doc_idx, start_position)
        self.payloads = {} # raw text stored, not tokens
        self.labels = {} # doc_idx to label mapping
        self._build_index(dataframe) #!!!!

    def _build_index(self, dataframe):
        for doc_idx, row in dataframe.iterrows():
            html = row['payload']
            label = row['label']

            # storing raw text, not tokens yet
            self.payloads[doc_idx] = html
            self.labels[doc_idx] = label

            # we estimate the chunk count per doc from the character length ~roughly 4 chars per token for HTML
            estimated_tokens = len(html) // 4
            n_chunks = max(1, len(range(0, estimated_tokens, self.stride)))

            for chunk_num in range(n_chunks):
                self.index.append((doc_idx, chunk_num))
        
        print("Index built. Total chunks: {len(self.index)}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        """We use __getitem__ to handle tokenizing and padding on the fly for each batch the trianing loop requests."""
        doc_idx, chunk_num = self.index[index] # Unpacking the tuple
        html = self.payloads[doc_idx]
        label = self.labels[doc_idx]

        # Estimate char window for this chunk
        chars_per_token = 4
        char_start = max(0, chunk_num * self.stride * chars_per_token - 256)
        char_end = char_start + (self.max_len * chars_per_token * 2)

        # Only tokenize relevant slice
        html_slice = html[char_start:char_end]

        # Tokenize the HTML content
        tokens = self.tokenizer(
            html_slice,
            return_tensors='pt',
            truncation=False,
            add_special_tokens=False # tokenize without special token first
        )['input_ids'][0]

        # Extract chunk from slice (adjusting for the character offset)
        chunk_start = min(
            chunk_num * self.stride - (char_start // chars_per_token), 
            0
        )
        chunk_start = max(0, chunk_start)
        
        # Leave room for <s> and </s> (RoBERTa requires exactly 2 special tokens)
        chunk = tokens[chunk_start:chunk_start + (self.max_len - 2)]

        if len(chunk) == 0:
            chunk = tokens[:(self.max_len - 2)]

        # put <s> at the start and </s> at the ned
        chunk = torch.cat([torch.tensor([0]), chunk, torch.tensor([2])])

        # Pad to max_len if shorter
        padding_length = self.max_len - len(chunk)
        attention_mask = torch.ones(len(chunk), dtype=torch.long) # ascribes 1s to 'real' tokens to pay attention to

        if padding_length > 0:
            chunk = torch.cat([
                chunk,
                torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(padding_length, dtype=torch.long) # ascribes 0s to padding tokens to ignore them in attention
            ])                                                # they contribute nothing to the CLS token's representation

        return {
            'input_ids': chunk,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long),
            'doc_idx': torch.tensor(doc_idx, dtype=torch.long)
        }

# ===== 3. Loss and train functions =====
# 3.1 Train function
def train_model(model, data_loader, optimizer, loss_fn, device, scheduler, n_examples):
    model = model.train()
    total_loss = 0
    correct_predictions = 0

    for batch in data_loader: 
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
    
    return correct_predictions.float() / n_examples, total_loss / len(data_loader)

# 3.2 Eval function
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval() 
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
    
    return correct_predictions.float() / n_examples, total_loss / len(data_loader)

# ===== 4. K-Fold Cross Validation =====
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def run_fold(train_df, test_df, fold_num):

    # 4.1 Class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['label'].values),
        y=train_df['label'].values
    )
    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    loss_fn = TriangularLoss(
        base_loss=nn.CrossEntropyLoss(weight=class_weight_tensor),
        num_classes=len(label_encoder.classes_)
    ).to(device)

    # 5. DataLoader
    train_dataset = ChunkedHTMLDataset(train_df.reset_index(drop=True), tokenizer, CONFIG['max_len'])
    test_dataset = ChunkedHTMLDataset(test_df.reset_index(drop=True), tokenizer, CONFIG['max_len'])

    # Data collator removed because padding is handled in the Dataset class now
    train_data_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False) 

    # 6. Model
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=len(label_encoder.classes_),
        use_safetensors=False
    ).to(device)

    for p in model.parameters():
        p.requires_grad = True

    # 7. Optimizer and scheduler
    total_steps = len(train_data_loader) * CONFIG['epochs']
    warmup_steps = int(0.1 * total_steps)
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # 8. Training loop
    for epoch in range(CONFIG['epochs']):
        print(f'Epoch {epoch + 1}/{CONFIG["epochs"]}')
        print('-' * 10)
        train_acc, train_loss = train_model(model, train_data_loader, optimizer, loss_fn, device, scheduler, len(train_dataset))
        print(f'Train loss {train_loss} accuracy {train_acc}')
        test_acc, test_loss = eval_model(model, test_data_loader, loss_fn, device, len(test_dataset))
        print(f'Test loss {test_loss} accuracy {test_acc}')

    # 9. Document-level evalution with loss (mean pooling of logits) and classification report
    doc_logits = defaultdict(list)
    doc_labels = {}
    total_test_loss = 0

    model.eval()
    with torch.no_grad():
        for batch in test_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_tensor = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Chunk-level loss on the GPU
            loss = loss_fn(outputs.logits, labels_tensor)
            total_test_loss += loss.item()

            #Moving data to CPU for aggregation and scikit learn metrics
            logits = outputs.logits.cpu().numpy()
            labels_cpu = labels_tensor.cpu().numpy()
            doc_indices = batch['doc_idx'].cpu().numpy()

            # Group logits and labels by document index
            for i in range(len(labels_cpu)):
                doc_idx = doc_indices[i]
                doc_logits[doc_idx].append(logits[i])
                doc_labels[doc_idx] = labels_cpu[i]

    # Calculate and print average chunk-level test loss
    avg_test_loss = total_test_loss / len(test_data_loader)
    print(f'Test loss (Chunk level): {avg_test_loss:.4f}')

    y_true_doc, y_pred_doc = [], []

    # Aggregate chunks back to documents using mean pooling
    for doc_idx in doc_logits.keys():
        pooled_logits = np.mean(doc_logits[doc_idx], axis=0)
        pred_label = np.argmax(pooled_logits)

        y_true_doc.append(doc_labels[doc_idx])
        y_pred_doc.append(pred_label)

    # Calculate final document-level metrics
    target_names_str = [str(cls) for cls in label_encoder.classes_]
    print(classification_report(y_true_doc, y_pred_doc, target_names=target_names_str, zero_division=1))

    macro_f1 = f1_score(y_true_doc, y_pred_doc, average='macro')

    # Save this fold's model
    save_path = f'./saved_models/{CONFIG["experiment_name"]}_fold{fold_num}'
    print(f"Saving model for fold {fold_num} at {save_path}")
    model.save_pretrained(save_path)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Aggregated document-level predictions for the confusion matrix
    return macro_f1, y_true_doc, y_pred_doc, save_path


# ===== Run SKF =====
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []
all_y_true = []  # collect across all folds
all_y_pred = []
best_f1 = 0
best_model_path = None

for fold, (train_idx, test_idx) in enumerate(skf.split(df['payload'], df['label'])):
    print(f"\n{'='*20} FOLD {fold+1}/5 {'='*20}")
    fold_train = df.iloc[train_idx]
    fold_test = df.iloc[test_idx]
    macro_f1, y_true, y_pred, save_path = run_fold(
        fold_train.reset_index(drop=True), # reset indices of test_df before passing it to ChunkedHTMLDataset to avoid weird indexing (45, 123, etc. instead of 0, 1,2)
        fold_test.reset_index(drop=True), 
        fold+1
        )
    
    fold_scores.append(macro_f1)
    all_y_true.extend(y_true)
    all_y_pred.extend(y_pred)
    
    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_model_path = save_path
    
    print(f"Fold {fold+1} macro F1: {macro_f1:.3f}")

print('List of possible accuracy:', fold_scores)
print('\nMaximum macro f1 That can be obtained from this model is:',
	max(fold_scores)*100, '%')
print('\nMinimum Accuracy:',
	min(fold_scores)*100, '%')
print('\nOverall Accuracy:',
	np.mean(fold_scores)*100, '%')
print('\nStandard Deviation is:', stdev(fold_scores))

print(f"\nMean macro F1: {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}")

print(f"\nBest model: {best_model_path} (F1: {best_f1:.3f})")


# ===== Confusion matrix on ALL folds combined =====
target_names_str = [str(cls) for cls in label_encoder.classes_]
cm = confusion_matrix(all_y_true, all_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names_str)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title(f'Confusion Matrix {CONFIG["experiment_name"]} (all folds)')
plt.tight_layout()
plt.savefig(f'confusion_matrix.{CONFIG["experiment_name"]}.png', dpi=150)
plt.show()

# ==== 10. Loss curve ====
#writer = SummaryWriter(f'runs/{CONFIG["experiment_name"]}')

#writer.add_scalar('Loss/train', train_loss, epoch)
#writer.add_scalar('Loss/test', test_loss, epoch)
#writer.add_scalar('Accuracy/train', train_acc, epoch)
#writer.add_scalar('Accuracy/test', test_acc, epoch)

#writer.close()

# view in tensorboard with: tensorboard --logdir=runs

