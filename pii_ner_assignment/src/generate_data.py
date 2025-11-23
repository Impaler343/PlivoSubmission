import json
import random
import os
from faker import Faker
from num2words import num2words

# Initialize Faker
fake = Faker()

# Configuration
OUTPUT_DIR = "data"
TRAIN_SIZE = 1000  # Requirement: 500-1000
DEV_SIZE = 200    # Requirement: 100-200
TEST_SIZE = 200   # Requirement: Unlabeled test set

# Templates grouped by Label
TEMPLATES = {
    "PERSON_NAME": [
        "my name is {val}", "this is {val} speaking", "contact {val}", 
        "name is {val}", "ask for {val}", "{val} is the manager",
        "meeting with {val}", "i am {val}"
    ],
    "EMAIL": [
        "email me at {val}", "send to {val}", "address is {val}", 
        "write to {val}", "contact {val}", "my email is {val}",
        "cc {val}"
    ],
    "PHONE": [
        "call me on {val}", "number is {val}", "dial {val}", 
        "reach me at {val}", "my phone is {val}", "mobile is {val}",
        "call {val}"
    ],
    "CREDIT_CARD": [
        "card number is {val}", "visa number {val}", "mastercard {val}", 
        "pay with {val}", "charge to {val}", "my card is {val}",
        "use card {val}"
    ],
    "DATE": [
        "born on {val}", "dob is {val}", "schedule for {val}", 
        "valid until {val}", "date is {val}", "on {val}"
    ],
    "CITY": [
        "live in {val}", "moving to {val}", "from {val}", 
        "office in {val}", "visit {val}", "city of {val}",
        "go to {val}"
    ],
    "LOCATION": [
        "located at {val}", "address is {val}", "meet at {val}", 
        "ship to {val}", "property at {val}", "live at {val}"
    ]
}

# Connectors to join multiple entities naturally
CONNECTORS = [
    " and ", " and ", " and ",  # High frequency
    " also ", " then ", " , ", " . ", " plus ",
    " whereas ", " but ", " ; "
]

# Filler text for suffixes or negative examples
FILLER_PHRASES = [
    "please", "thank you", "as soon as possible", "if you can", "over", 
    "is that correct", "did you get that", "let me know", "right now",
    "have a nice day", "bye", "okay", "confirmed", "verified"
]

# Negative examples (Sentences with NO entities)
NEGATIVE_SENTENCES = [
    "hello how are you", "can you hear me", "i will be there soon", 
    "what is the time", "the weather is nice today", "i need to go to the store",
    "thank you very much", "please wait a moment", "i am not sure about that",
    "let us meet later", "do you have the documents", "turn left at the corner",
    "this is a test message", "just checking in", "good morning", 
    "could you repeat that", "i did not understand", "hold on a second",
    "where are we going", "that sounds good to me"
]

def noise_text(text):
    """Applies STT noise: lowercase, remove punctuation."""
    return text.lower().replace("-", " ").replace(",", "").replace(".", "").replace("?", "")

def noise_email(email):
    # Split email to noise specific parts
    try:
        user, domain = email.split('@')
        if random.random() > 0.5:
            return f"{user} at {domain}".replace(".", " dot ").lower()
        return email.replace("@", " at ").replace(".", " dot ").lower()
    except:
        return email.replace("@", " at ").replace(".", " dot ").lower()

def noise_phone(phone):
    digits = ''.join(filter(str.isdigit, phone))
    # 80% chance to spell out digits (typical STT)
    if random.random() < 0.8:
        return " ".join([num2words(int(d)) for d in digits])
    return digits

def generate_single_component(label):
    """Generates a single entity phrase (text, start, end, label)."""
    
    # Generate Value
    raw_val = ""
    if label == "PERSON_NAME": raw_val = fake.name()
    elif label == "EMAIL": raw_val = fake.email()
    elif label == "PHONE": raw_val = fake.phone_number()
    elif label == "CREDIT_CARD": raw_val = fake.credit_card_number()
    elif label == "DATE": raw_val = str(fake.date())
    elif label == "CITY": raw_val = fake.city()
    elif label == "LOCATION": raw_val = fake.street_address()
    
    # Apply Noise
    if label == "EMAIL": noised_val = noise_email(raw_val)
    elif label == "PHONE" or label == "CREDIT_CARD": noised_val = noise_phone(raw_val)
    else: noised_val = noise_text(raw_val)
    
    # Select Template
    template = random.choice(TEMPLATES[label])
    prefix, suffix_part = template.split("{val}")
    
    # Construct partial text
    # We strip prefix to ensure we don't have double spaces when connecting later
    clean_prefix = prefix.lstrip() 
    
    partial_text = f"{clean_prefix}{noised_val}{suffix_part}"
    
    # Calculate offset within this snippet
    # If prefix was "my name is ", start is len("my name is ")
    start_index = len(clean_prefix)
    end_index = start_index + len(noised_val)
    
    return {
        "text": partial_text,
        "local_start": start_index,
        "local_end": end_index,
        "label": label
    }

def generate_example(uid):
    # 1. Negative Example (20% chance)
    if random.random() < 0.2:
        text = random.choice(NEGATIVE_SENTENCES)
        if random.random() > 0.5: text += " " + random.choice(FILLER_PHRASES)
        return {"id": f"utt_{uid}", "text": noise_text(text), "entities": []}

    # 2. Determine number of entities (1 to 3)
    # Weights favour 1 or 2 entities
    num_entities = random.choices([1, 2, 3], weights=[0.5, 0.4, 0.1])[0]
    
    components = []
    used_labels = set()
    
    for _ in range(num_entities):
        # Try to pick a label we haven't used in this sentence yet
        available_labels = list(set(TEMPLATES.keys()) - used_labels)
        if not available_labels: available_labels = list(TEMPLATES.keys())
        
        label = random.choice(available_labels)
        used_labels.add(label)
        
        components.append(generate_single_component(label))
    
    # 3. Stitch components together
    full_text = ""
    final_entities = []
    
    for i, comp in enumerate(components):
        connector = ""
        if i > 0:
            connector = random.choice(CONNECTORS)
        
        # Calculate current cursor position
        current_offset = len(full_text) + len(connector)
        
        # Append text
        full_text += connector + comp['text']
        
        # Adjust entity offsets relative to full text
        final_entities.append({
            "start": current_offset + comp['local_start'],
            "end": current_offset + comp['local_end'],
            "label": comp['label']
        })
        
    # 4. Add optional filler at the end
    if random.random() > 0.7:
        full_text += " " + random.choice(FILLER_PHRASES)

    return {
        "id": f"utt_{uid}",
        "text": full_text.strip(), # Final strip just in case
        "entities": final_entities
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating Multi-Entity Data...")
    
    # Train/Dev with labels
    train = [generate_example(i) for i in range(TRAIN_SIZE)]
    dev = [generate_example(i + TRAIN_SIZE) for i in range(DEV_SIZE)]
    
    # Test without labels
    test_labeled = [generate_example(i + TRAIN_SIZE + DEV_SIZE) for i in range(TEST_SIZE)]
    test = [{"id": item["id"], "text": item["text"]} for item in test_labeled]
    
    with open(f"{OUTPUT_DIR}/train.jsonl", "w") as f:
        for item in train: f.write(json.dumps(item) + "\n")
        
    with open(f"{OUTPUT_DIR}/dev.jsonl", "w") as f:
        for item in dev: f.write(json.dumps(item) + "\n")

    with open(f"{OUTPUT_DIR}/test.jsonl", "w") as f:
        for item in test: f.write(json.dumps(item) + "\n")
        
    print(f"Done. Generated {TRAIN_SIZE} train, {DEV_SIZE} dev, {TEST_SIZE} test.")

if __name__ == "__main__":
    main()