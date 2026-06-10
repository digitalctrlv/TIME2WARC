import pandas as pd
import torch
import json
import pickle
import numpy as np
import warnings
from pathlib import Path
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from labeling_function_v8 import SignalExtractor

# environment specific configuration
from env import *

# ===== Device settings =====
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Inference running on {device}")

# ===== 0. Masking & Encoder =====
def mask_payload(payload):
    if not isinstance(payload, str):
        return ""
    for signal_name, pattern in SignalExtractor.regex_patterns.items():
        payload = pattern.sub("", payload)
    return payload

# Load the label encoder saved during training
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# ===== 1. Load best model =====
best_model_path = './saved_models/YOUR_BEST_MODEL_fold3' # <-- Change this actual best fold path
model = RobertaForSequenceClassification.from_pretrained(Path(best_model_path))
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = model.to(device)
model.eval()

# ===== 2. Load inference dataset =====
df_inference = pd.read_json(config.locations['inference'], lines=True)

# Apply same masking as training
df_inference['payload'] = df_inference['payload'].apply(mask_payload)

# ===== 3. Predict exactly as trained (No chunking) =====
def predict_document(html, tokenizer, model, device, max_len=512):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        encoding = tokenizer(
            text=html,
            max_length=max_len,
            padding=True, 
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits

    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_label = int(np.argmax(probs))

    return pred_label, probs

# ===== 4. Run inference =====
predicted_labels = []
predicted_periods = []
confidence_scores = []

for idx, row in df_inference.iterrows():
    if idx % 100 == 0:
        print(f"Processing {idx}/{len(df_inference)}...")

    pred_label, probs = predict_document(
        row['payload'], tokenizer, model, device
    )

    predicted_labels.append(pred_label)
    predicted_periods.append(
        label_encoder.inverse_transform([pred_label])[0] 
    )
    confidence_scores.append(float(np.max(probs)))

# ===== 5. Annotate dataframe =====
df_inference['predicted_period'] = predicted_periods
df_inference['predicted_period_confidence'] = confidence_scores

# ===== 6. Export as JSONL =====
output_path = './output/inference_annotated.jsonl'
Path('./output').mkdir(exist_ok=True)

df_inference.to_json(
    output_path,
    orient='records',
    lines=True,
    force_ascii=False
)

print(f"\nSaved {len(df_inference)} annotated records to {output_path}")
print(f"\nPredicted period distribution:")
print(df_inference['predicted_period'].value_counts())
print(f"\nMean confidence: {df_inference['predicted_period_confidence'].mean():.3f}")

# ==== !!!! POST-PREDICTION ANALYSIS & FILTERING !!!! ====
CONFIDENCE_THRESHOLD = 0.6

df_inference['predicted_period_final'] = df_inference.apply(
    lambda row: row['predicted_period'] 
    if row['predicted_period_confidence'] >= CONFIDENCE_THRESHOLD 
    else 'uncertain',
    axis=1
)

print(f"\nHigh confidence predictions: "
      f"{(df_inference['predicted_period_confidence'] >= CONFIDENCE_THRESHOLD).sum()}"
      f"/{len(df_inference)}")
print(f"Uncertain: "
      f"{(df_inference['predicted_period_final'] == 'uncertain').sum()}")