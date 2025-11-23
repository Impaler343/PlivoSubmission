import json
import argparse
import torch
import os
import re
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii

# =============================================================================
# CONFIGURATION: Dynamic Thresholds & Regex Patterns
# =============================================================================

THRESHOLDS = {
    "PHONE": 0.15,       # Aggressive recall for numbers
    "CREDIT_CARD": 0.15, # Aggressive recall for numbers
    "DATE": 0.20,        # Dates can be fuzzy, lower threshold
    "EMAIL": 0.40,       # Keep high (easy to detect)
    "PERSON_NAME": 0.40, # Keep high
    "CITY": 0.40,
    "LOCATION": 0.40,
    "DEFAULT": 0.50
}

def apply_regex_rescue(text, current_entities):
    """
    AHA MOMENT: The Final Safety Net.
    Scans the text for strict patterns (Dates, Raw Phones) that the model 
    might have missed due to low confidence or noise.
    """
    new_entities = list(current_entities)
    existing_ranges = set()
    
    # Mark tokens already covered by the model as 'occupied'
    for e in current_entities:
        for i in range(e['start'], e['end']):
            existing_ranges.add(i)

    date_pattern = re.compile(
        r'\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)(?:\s+\d{4})?\b',
        re.IGNORECASE
    )
    
    for match in date_pattern.finditer(text):
        start, end = match.span()
        # Only add if it doesn't overlap with existing prediction
        if not any(i in existing_ranges for i in range(start, end)):
            new_entities.append({
                "start": start,
                "end": end,
                "label": "DATE",
                "pii": True
            })
            # Mark as occupied so we don't add overlapping entities later
            for i in range(start, end): existing_ranges.add(i)

    # 2. Regex for Raw Phones: 10 digits starting with 6-9 (Common format in test cases)
    # Also generic 10 digits: \b\d{10}\b
    phone_pattern = re.compile(r'\b[6-9]\d{9}\b')
    for match in phone_pattern.finditer(text):
        start, end = match.span()
        if not any(i in existing_ranges for i in range(start, end)):
            new_entities.append({
                "start": start,
                "end": end,
                "label": "PHONE",
                "pii": True
            })
            for i in range(start, end): existing_ranges.add(i)
            
    # 3. Regex for Spoken Email (fallback): "name at domain dot com"
    email_pattern = re.compile(r'\b\w+\s+at\s+\w+\s+dot\s+\w+\b', re.IGNORECASE)
    for match in email_pattern.finditer(text):
        start, end = match.span()
        if not any(i in existing_ranges for i in range(start, end)):
             new_entities.append({
                "start": start,
                "end": end,
                "label": "EMAIL",
                "pii": True
            })

    return new_entities

def validate_span(text, label):
    """
    Heuristics to filter obvious False Positives.
    """
    text = text.lower().strip()
    
    if len(text) < 2: return False

    if label == "EMAIL":
        if not any(x in text for x in ["at", "@", "dot", "."]): return False
            
    if label == "PHONE":
        if len(text) < 3: return False # Allow short for gap repair, but filter garbage
            
    if label == "CREDIT_CARD":
        if len(text) < 4: return False
            
    if label == "DATE":
        if len(text) < 3: return False

    return True

def repair_gaps(spans, text, max_gap_chars=3):
    """
    Merges fragmented spans. Essential for 'spaced out' numbers.
    Ex: [555] + " " + [0199] -> [555 0199]
    """
    if not spans: return []
    sorted_spans = sorted(spans, key=lambda x: x[0])
    merged = [sorted_spans[0]]
    
    for current in sorted_spans[1:]:
        prev = merged[-1]
        prev_s, prev_e, prev_l = prev
        curr_s, curr_e, curr_l = current
        
        gap_text = text[prev_e:curr_s]
        
        # Merge if Label Matches AND Gap is small
        if prev_l == curr_l and len(gap_text) <= max_gap_chars:
            merged[-1] = (prev_s, curr_e, prev_l)
        else:
            merged.append(current)
    return merged

def bio_to_spans_strict(offsets, label_ids):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        if start == 0 and end == 0: continue
        label_str = ID2LABEL.get(int(lid), "O")
        
        if label_str == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        prefix, ent_type = label_str.split("-", 1)

        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
            
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                    current_label = None

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans

def mc_dropout_predict(model, input_ids, attention_mask, num_samples=3):
    """
    Monte Carlo Dropout Inference.
    """
    model.train() # Enable dropout
    all_logits = []
    with torch.no_grad():
        for _ in range(num_samples):
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(out.logits[0])
    return torch.stack(all_logits).mean(dim=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    
    # Use Full Precision (FP32) for max accuracy
    model.to(args.device)

    results = {}
    MC_SAMPLES = 3

    print(f"Inference: MC_Dropout={MC_SAMPLES}, Thresholds=Dynamic, Regex=Enabled.")

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            # 1. MC Dropout Prediction
            logits = mc_dropout_predict(model, input_ids, attention_mask, num_samples=MC_SAMPLES)
            
            # 2. Per-Class Thresholding Logic
            probs = F.softmax(logits, dim=-1)
            confidences, pred_ids = torch.max(probs, dim=-1)
            
            clean_preds = []
            for pid, conf in zip(pred_ids.tolist(), confidences.tolist()):
                label_str = ID2LABEL.get(pid, "O")
                if "-" in label_str:
                    entity_type = label_str.split("-")[1]
                else:
                    entity_type = "DEFAULT"
                
                req_threshold = THRESHOLDS.get(entity_type, THRESHOLDS["DEFAULT"])
                
                if pid != 0 and conf < req_threshold:
                    clean_preds.append(0) 
                else:
                    clean_preds.append(pid)

            # 3. Decode
            raw_spans = bio_to_spans_strict(offsets, clean_preds)
            
            # 4. Repair Gaps
            repaired_spans = repair_gaps(raw_spans, text)
            
            # 5. Build Initial Entity List
            ents = []
            for s, e, lab in repaired_spans:
                span_text = text[s:e]
                if validate_span(span_text, lab):
                    ents.append({
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    })
            
            ents = apply_regex_rescue(text, ents)
                    
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions to {args.output}")

if __name__ == "__main__":
    main()