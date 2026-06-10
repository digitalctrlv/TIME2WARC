import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import stdev
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
from labeling_function_v8 import SignalExtractor

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
    'experiment_name': 'v2.3_optimal_weights',
    'max_len': 512,
    'batch_size': 4,
    'epochs': 10,
    'freeze': False,
    'learning_rate': 2e-5,
    #'dropout': 0.4,
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

        encoding = self.tokenizer(
            text=html_string,
            max_length=self.max_len,
            padding=False,
            return_attention_mask=True,
            return_tensors='pt',
 
           truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ===== 3. Loss and Train functions =====
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
    train_data_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=data_collator)
    test_data_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=data_collator)

    # 6. Model
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=len(label_encoder.classes_),
        use_safetensors=False
    ).to(device)

    for p in model.parameters():
        p.requires_grad = True

    # 7. Optimizer and Scheduler
    total_steps = len(train_data_loader) * CONFIG['epochs']
    warmup_steps = int(0.1 * total_steps)
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Initialize tracking variables for model saving
    best_val_loss = float('inf')
    save_path = f'./saved_models/{CONFIG["experiment_name"]}_fold{fold_num}'

    # 8. Training Loop
    for epoch in range(CONFIG['epochs']):
        print(f'Epoch {epoch + 1}/{CONFIG["epochs"]}')
        print('-' * 10)
        
        train_acc, train_loss = train_model(model, train_data_loader, optimizer, loss_fn, device, scheduler, len(train_df))
        print(f'Train loss: {train_loss:.4f} | accuracy: {train_acc:.4f}')
        
        test_acc, test_loss = eval_model(model, test_data_loader, loss_fn, device, len(test_df))
        print(f'Test loss: {test_loss:.4f} | accuracy: {test_acc:.4f}')

        # Save best model based on validation loss
        if test_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.4f} --> {test_loss:.4f}). Saving model...")
            best_val_loss = test_loss
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path) 
        else:
            print(f"Validation loss did not improve.")

    # 9. Evaluation
    # Load the best model weights back into memory before final evaluation
    print(f"Loading best model for fold {fold_num} evaluation...")
    del model
    torch.cuda.empty_cache()
    model = RobertaForSequenceClassification.from_pretrained(save_path).to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            y_true.extend(batch['labels'].cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    target_names_str = [str(cls) for cls in label_encoder.classes_]
    print(classification_report(y_true, y_pred, target_names=target_names_str, zero_division=1))

    print(f"\nStandard accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"Windowed accuracy (+-1): {window_accuracy(y_true, y_pred):.3f}\n")

    macro_f1 = f1_score(y_true, y_pred, average='macro')

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return macro_f1, y_true, y_pred, save_path


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

print(f"\nMean macro F1: {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}")
print(f"\nBest model: {best_model_path} (F1: {best_f1:.3f})")

print(f"\nList of possible accuracy:', fold_scores")
print("\nMaximum accuracy that can be obtained from this model is:", max(fold_scores)*100, "%")
print("\nMinimum accuracy:", min(fold_scores)*100, "%")
print("\nStandard deviation is:", stdev(fold_scores))

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

