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
from statistics import stdev
from transformers import RobertaForSequenceClassification, RobertaTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm.notebook import tqdm
from dlordinal.losses import TriangularLoss
from labeling_function_v6 import SignalExtractor
from collections import defaultdict 
import warnings

# environment specific configuration
from env import *

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
   print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB \n")
   print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB \n")
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
    'experiment_name': 'v4_kfold_slidwind',
    'max_len': 512,
    'batch_size': 4,
    'epochs': 10,
    'freeze': False,
    'learning_rate': 2e-5,
    # 'dropout': 0.4,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1
}

# ===== 1. Load data =====
df = pd.read_json(config.locations['train_w13'], lines=True)

def mask_payload(payload):
    if not isinstance(payload, str):
        return ""

    # Loop over reg ex dictionary from the labeling functions script to blank out signals
    for signal_name, pattern in SignalExtractor.regex_patterns.items():
        payload = pattern.sub("", payload)
    return payload

df['payload'] = df['payload'].apply(mask_payload)

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['period_bucket'])

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ===== 2. Dataset and Tokenizer =====
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

class HTMLDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.payload = dataframe['payload'].values
        self.labels = dataframe['label'].values
        self.tokenizer = tokenizer 
        self.max_len = max_len 
    
    def __len__(self):
        return len(self.payload)
    
    def __getitem__(self, index):
        html_string = self.payload[index]
        label = self.labels[index]

        tokens = self.tokenizer(
            text=html_string,
            max_length=self.max_len,
            padding=False,
            return_attention_mask=True,
            return_tensors='pt',
 
           truncation=False
        )['input_ids'][0]

        # smart truncation
        if len(tokens) > self.max_len:
            half = self.max_len // 2
            tokens = torch.cat([tokens[:half], tokens[-half:]])

        # manual padding
        padding_length = self.max_len - len(tokens)
        attention_mask = torch.ones(len(tokens), dtype=torch.long)

        if padding_length > 0:
            tokens = torch.cat([tokens, torch.zeros(padding_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])

        return {
            'input_ids': tokens,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
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

# 3.3 Windowed accuracy function
def window_accuracy(y_true, y_pred, window=1):
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    correct = np.sum(np.abs(y_true_arr - y_pred_arr) <= window)
    return correct / len(y_true_arr)

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
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataset = HTMLDataset(train_df.reset_index(drop=True), tokenizer, CONFIG['max_len'])
    test_dataset = HTMLDataset(test_df.reset_index(drop=True), tokenizer, CONFIG['max_len'])
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
        train_acc, train_loss = train_model(model, train_data_loader, optimizer, loss_fn, device, scheduler, len(train_df))
        print(f'Train loss {train_loss} accuracy {train_acc}')
        test_acc, test_loss = eval_model(model, test_data_loader, loss_fn, device, len(test_df))
        print(f'Test loss {test_loss} accuracy {test_acc}')

    # 9. Document-level evaluation with sliding window
    doc_logits = defaultdict(list)
    doc_labels = {}
    total_test_loss = 0

    model.eval()
    with torch.no_grad():
        # First compute chunk-level test loss using test_data_loader as before
        for batch in test_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_tensor = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels_tensor)
            total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_data_loader)
        print(f'Test loss (chunk level): {avg_test_loss:.4f}')

        # Then do sliding window evaluation row by row for document-level predictions
        for doc_idx, row in test_df.reset_index(drop=True).iterrows():
            label = row['label']
            html = row['payload']

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tokens = tokenizer(
                    html,
                    truncation=False,
                    return_tensors='pt'
                )['input_ids'][0]

            for start in range(0, len(tokens), 256):
                chunk = tokens[start:start + 512]
                pad_len = 512 - len(chunk)
                attention_mask = torch.ones(len(chunk), dtype=torch.long)

                if pad_len > 0:
                    chunk = torch.cat([chunk, torch.zeros(pad_len, dtype=torch.long)])
                    attention_mask = torch.cat([attention_mask, 
                                               torch.zeros(pad_len, dtype=torch.long)])

                outputs = model(
                    input_ids=chunk.unsqueeze(0).to(device),
                    attention_mask=attention_mask.unsqueeze(0).to(device)
                )
                doc_logits[doc_idx].append(outputs.logits.cpu().numpy())

            doc_labels[doc_idx] = label

    # Aggregate chunks → document prediction
    y_true_doc, y_pred_doc = [], []
    for doc_idx in doc_logits.keys():
        pooled = np.mean(doc_logits[doc_idx], axis=0)
        y_true_doc.append(doc_labels[doc_idx])
        y_pred_doc.append(int(np.argmax(pooled)))

    target_names_str = [str(cls) for cls in label_encoder.classes_]
    print(classification_report(y_true_doc, y_pred_doc, 
                                target_names=target_names_str, zero_division=1))
    
    print(f"\nStandard accuracy: {accuracy_score(y_true_doc, y_pred_doc):.3f}")
    print(f"Windowed accuracy (+-1): {window_accuracy(y_true_doc, y_pred_doc):.3f}\n")

    macro_f1 = f1_score(y_true_doc, y_pred_doc, average='macro')

    # Save model
    save_path = f'./saved_models/{CONFIG["experiment_name"]}_fold{fold_num}'
    print(f"Saving model for fold {fold_num} at {save_path}\n")
    model.save_pretrained(save_path)

    del model
    gc.collect()
    torch.cuda.empty_cache()

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
    macro_f1, y_true, y_pred, save_path = run_fold(fold_train, fold_test, fold+1)
    
    fold_scores.append(macro_f1)
    all_y_true.extend(y_true)
    all_y_pred.extend(y_pred)
    
    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_model_path = save_path
    
    print(f"Fold {fold+1} macro F1: {macro_f1:.3f}")

print('List of possible accuracy:', fold_scores)
print('\nMaximum Accuracy That can be obtained from this model is:',
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

