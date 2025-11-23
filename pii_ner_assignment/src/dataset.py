import json
import random
import os
from faker import Faker
from num2words import num2words

fake = Faker()

OUTPUT_DIR = "data"
TRAIN_SIZE = 2000
DEV_SIZE = 200    
TEST_SIZE = 200   

# ==========================================
# 1. TEMPLATES
# ==========================================
TEMPLATES = {
    "PERSON_NAME": [
        "my name is {val}", "this is {val}", "i am {val}", "name is {val}", 
        "speaking with {val}", "call me {val}", "ask for {val}", 
        "it is {val}", "identifying as {val}"
    ],
    "EMAIL": [
        "reach me at {val}", "email me at {val}", "my email is {val}", 
        "send to {val}", "contact me at {val}", "address is {val}", 
        "email is {val}", "at {val}", "forward to {val}"
    ],
    "PHONE": [
        "phone is {val}", "call me on {val}", "call {val}", "dial {val}", 
        "number is {val}", "mobile is {val}", "reach me at {val}", 
        "contact {val}", "at {val}", "cell is {val}"
    ],
    "CREDIT_CARD": [
        "card number {val}", "credit card {val}", "number is {val}", 
        "visa {val}", "mastercard {val}", "pay with {val}", "use {val}",
        "billing to {val}", "card {val}"
    ],
    "DATE": [
        "on {val}", "date {val}", "born {val}", "schedule for {val}", 
        "deadline {val}", "from {val}", "until {val}", "by {val}",
        "meeting on {val}", "starting {val}"
    ],
    "CITY": [
        # AHA AUGMENTATION: "i will be in..."
        "i will be in {val}", "will be in {val}", "be in {val}", 
        "going to {val}", "visit {val}", "live in {val}", "from {val}", 
        "city {val}", "office in {val}", "staying in {val}"
    ],
    "LOCATION": [
        "located at {val}", "address {val}", "near {val}", 
        "live at {val}", "ship to {val}", "property {val}", "house at {val}"
    ]
}

# AHA AUGMENTATION: Added " " (space) so entities can flow directly 
# e.g., "in Delhi on 15 August" instead of "in Delhi AND on 15 August"
CONNECTORS = [
    " ", " ", " ", # High weight for direct connection
    " and ", " also ", " then ", " , ", " . ", " but ", " ; ", " or "
]

FILLER_PHRASES = [
    "please", "thank you", "as soon as possible", "right now",
    "have a nice day", "bye", "okay", "confirmed"
]

NEGATIVE_SENTENCES = [
    "hello how are you", "can you hear me", "i will be there soon", 
    "what is the time", "i need to go to the store", "thank you very much", 
    "please wait a moment", "let us meet later", "turn left at the corner"
]

# ==========================================
# 2. VALUE GENERATORS
# ==========================================
def generate_spoken_date():
    dt = fake.date_object()
    formats = [
        "%d %B %Y",     # 15 August 2025 (Target Format)
        "%d %B %Y",     # Boost probability
        "%d %b %Y",     # 15 Aug 2025
        "%B %d %Y",     # August 15 2025
        "%Y-%m-%d",     # 2025-08-15
        "%d %B"         # 15 August
    ]
    return dt.strftime(random.choice(formats)).lower()

def generate_raw_phone():
    if random.random() < 0.5:
        return fake.phone_number()
    return f"{random.randint(6,9)}{fake.numerify('#########')}"

# ==========================================
# 3. NOISE FUNCTIONS
# ==========================================
def noise_fragment_digits(text):
    if not text: return text
    chars = list(text)
    noisy_chars = []
    for c in chars:
        noisy_chars.append(c)
        if c.isdigit() and random.random() < 0.4:
            noisy_chars.append(" ")
    return "".join(noisy_chars).replace("  ", " ")

def noise_text(text):
    return text.lower().replace("-", " ").replace(",", "").replace(".", "").replace("?", "").replace("!", "")

def noise_email(email):
    try:
        user, domain = email.split('@')
        if random.random() < 0.8:
            return f"{user} at {domain}".replace(".", " dot ").lower()
        return email.replace("@", " at ").replace(".", " dot ").lower()
    except:
        return email
        
def noise_phone(phone):
    digits = ''.join(filter(str.isdigit, phone))
    rand = random.random()
    if rand < 0.2:
        return " ".join([num2words(int(d)) for d in digits])
    elif rand < 0.6:
        return noise_fragment_digits(digits)
    else:
        return digits

def noise_credit_card(cc):
    digits = ''.join(filter(str.isdigit, cc))
    rand = random.random()
    if rand < 0.2:
        prefix = " ".join([num2words(int(d)) for d in digits[:4]])
        suffix = digits[4:]
        return f"{prefix} {noise_fragment_digits(suffix)}"
    elif rand < 0.7:
        return noise_fragment_digits(digits)
    else:
        return digits

def generate_single_component(label):
    raw_val = ""
    if label == "PERSON_NAME": raw_val = fake.name()
    elif label == "EMAIL": raw_val = fake.email()
    elif label == "PHONE": raw_val = generate_raw_phone()
    elif label == "DATE": raw_val = generate_spoken_date()
    elif label == "CREDIT_CARD": raw_val = fake.credit_card_number()
    elif label == "CITY": raw_val = fake.city()
    elif label == "LOCATION": raw_val = fake.street_address()
    
    # Apply Noise
    if label == "EMAIL": noised_val = noise_email(raw_val)
    elif label == "PHONE": noised_val = noise_phone(raw_val)
    elif label == "CREDIT_CARD": noised_val = noise_credit_card(raw_val)
    elif label == "DATE": noised_val = noise_text(raw_val)
    else: noised_val = noise_text(raw_val)
    
    template = random.choice(TEMPLATES[label])
    prefix, suffix_part = template.split("{val}")
    clean_prefix = noise_text(prefix).lstrip() 
    clean_suffix = noise_text(suffix_part).rstrip()
    if clean_prefix and not clean_prefix.endswith(" "): clean_prefix += " "
        
    partial_text = f"{clean_prefix}{noised_val}{clean_suffix}"
    start = len(clean_prefix)
    end = start + len(noised_val)
    
    return {"text": partial_text, "local_start": start, "local_end": end, "label": label}

def generate_example(uid):
    if random.random() < 0.2:
        text = random.choice(NEGATIVE_SENTENCES)
        if random.random() > 0.5: text += " " + random.choice(FILLER_PHRASES)
        return {"id": f"utt_{uid}", "text": noise_text(text), "entities": []}

    num = random.choices([1, 2, 3], weights=[0.4, 0.5, 0.1])[0]
    components = []
    used = set()
    for _ in range(num):
        avail = list(set(TEMPLATES.keys()) - used)
        if not avail: avail = list(TEMPLATES.keys())
        lbl = random.choice(avail)
        used.add(lbl)
        components.append(generate_single_component(lbl))
    
    full_text = ""
    ents = []
    for i, comp in enumerate(components):
        conn = ""
        if i > 0: conn = random.choice(CONNECTORS)
        
        # Fix: If connector is just space, ensure we don't double space if prev text ended with space
        if conn == " " and full_text.endswith(" "):
            conn = ""
            
        curr_off = len(full_text) + len(conn)
        full_text += conn + comp['text']
        ents.append({"start": curr_off + comp['local_start'], "end": curr_off + comp['local_end'], "label": comp['label']})
        
    if random.random() > 0.7: full_text += " " + random.choice(FILLER_PHRASES)
    return {"id": f"utt_{uid}", "text": full_text.strip(), "entities": ents}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating AUGMENTED Data (Direct Flows)...")
    train = [generate_example(i) for i in range(TRAIN_SIZE)]
    dev = [generate_example(i + TRAIN_SIZE) for i in range(DEV_SIZE)]
    with open(f"{OUTPUT_DIR}/train.jsonl", "w") as f:
        for item in train: f.write(json.dumps(item) + "\n")
    with open(f"{OUTPUT_DIR}/dev.jsonl", "w") as f:
        for item in dev: f.write(json.dumps(item) + "\n")
    print(f"Done.")

if __name__ == "__main__":
    main()