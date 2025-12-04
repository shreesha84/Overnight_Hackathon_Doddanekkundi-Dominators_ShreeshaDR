import pandas as pd
import random
import uuid
from datetime import datetime, timedelta

# --- CONFIGURATION ---
NUM_ROWS = 50000          # Size of dataset
NUM_FRAUDSTERS = 50       # Number of bad actors
START_DATE = datetime(2024, 1, 1, 8, 0, 0)

# --- HELPER FUNCTIONS ---
def get_random_upi():
    banks = ["@oksbi", "@okhdfc", "@ybl", "@paytm", "@ibl"]
    names = ["user", "shop", "merchant", "kiosk", "service"]
    return f"{random.choice(names)}_{random.randint(1000, 99999)}{random.choice(banks)}"

def get_location():
    return random.choice(["Mumbai", "Delhi", "Bangalore", "Pune", "Chennai", "Kolkata", "Hyderabad"])

print(f"Generating {NUM_ROWS} transactions... This may take 10 seconds.")

# 1. Generate Normal "Boring" Data
data = []
for _ in range(NUM_ROWS):
    txn = {
        "Transaction_ID": str(uuid.uuid4())[:12],
        "Sender_ID": get_random_upi(),
        "Receiver_ID": get_random_upi(),
        "Amount": round(random.uniform(10, 5000), 2),
        "Timestamp": START_DATE + timedelta(seconds=random.randint(0, 86400)), # Random time in 24 hrs
        "Location": get_location(),
        "Type": "P2P"
    }
    data.append(txn)

# 2. INJECT FRAUD RING 1: "The Spider Web" (One guy receiving money from 50 people instantly)
# This simulates a "Lottery Scam" or "Money Mule"
mule_account = "scammer_mule@ybl"
base_time = START_DATE + timedelta(hours=14) # 2 PM

for _ in range(50):
    data.append({
        "Transaction_ID": f"FRAUD_WEB_{random.randint(100,999)}",
        "Sender_ID": get_random_upi(), # Different victims
        "Receiver_ID": mule_account,   # SAME receiver
        "Amount": 500,                 # Same small amount
        "Timestamp": base_time + timedelta(seconds=random.randint(0, 300)), # All within 5 mins
        "Location": "Delhi",
        "Type": "P2P"
    })

# 3. INJECT FRAUD RING 2: "The Location Jumper"
# User pays in Mumbai, then 5 mins later pays in Delhi (Impossible speed)
jumper_user = "fast_thief@oksbi"
jump_time = START_DATE + timedelta(hours=10)

data.append({
    "Transaction_ID": "FRAUD_JUMP_1",
    "Sender_ID": jumper_user, "Receiver_ID": "shop_A@paytm",
    "Amount": 2000, "Timestamp": jump_time, "Location": "Mumbai", "Type": "P2M"
})
data.append({
    "Transaction_ID": "FRAUD_JUMP_2",
    "Sender_ID": jumper_user, "Receiver_ID": "shop_B@paytm",
    "Amount": 8000,
    "Timestamp": jump_time + timedelta(minutes=5), # 5 mins later
    "Location": "Delhi", # Impossible jump
    "Type": "P2M"
})

# 4. Save to CSV
df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True) # Shuffle everything
df.to_csv("large_upi_dataset.csv", index=False)

print(" Success! File 'large_upi_dataset.csv' created with 50,000+ rows.")
print(f" Bad Actor to find: {mule_account}")