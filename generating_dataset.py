import pandas as pd
import random
import uuid
from datetime import datetime, timedelta

# --- CONFIGURATION ---
NUM_ROWS = 5000          # Total transactions
START_DATE = datetime(2024, 1, 1, 8, 0, 0)

# --- HELPER LISTS ---
banks = ["@oksbi", "@okhdfc", "@ybl", "@paytm", "@ibl"]
names = ["arjun", "priya", "rahul", "sneha", "vikram", "amit", "pooja", "karan", "simran"]
locations = ["Mumbai", "Delhi", "Bangalore", "Pune", "Chennai", "Kolkata", "Hyderabad"]

def get_random_upi():
    return f"{random.choice(names)}_{random.randint(100, 9999)}{random.choice(banks)}"

print(f"Generating {NUM_ROWS} transactions with MULTIPLE fraud rings...")

data = []

# 1. GENERATE NORMAL "NOISE" DATA
for _ in range(NUM_ROWS):
    data.append({
        "Transaction_ID": str(uuid.uuid4())[:12],
        "Sender_ID": get_random_upi(),
        "Receiver_ID": get_random_upi(),
        "Amount": round(random.uniform(10, 3000), 2),
        "Timestamp": START_DATE + timedelta(seconds=random.randint(0, 86400)), 
        "Location": random.choice(locations)
    })

# 2. INJECT FRAUD RING 1: "The Big Boss" (High Volume)
# Receives 50 payments (Massive red node)
boss_mule = "KINGPIN_SCAM@ybl"
base_time = START_DATE + timedelta(hours=12) 

for _ in range(50):
    data.append({
        "Transaction_ID": f"RING1_{random.randint(1000,9999)}",
        "Sender_ID": get_random_upi(),
        "Receiver_ID": boss_mule,
        "Amount": 1000,
        "Timestamp": base_time + timedelta(seconds=random.randint(0, 600)),
        "Location": "Mumbai"
    })

# 3. INJECT FRAUD RING 2: "The Phishing Bot" (Medium Volume)
# Receives 25 payments (Medium red node)
bot_mule = "PHISHING_BOT@oksbi"
for _ in range(25):
    data.append({
        "Transaction_ID": f"RING2_{random.randint(1000,9999)}",
        "Sender_ID": get_random_upi(),
        "Receiver_ID": bot_mule,
        "Amount": 150, # Small amounts
        "Timestamp": base_time + timedelta(minutes=random.randint(0, 120)),
        "Location": "Delhi"
    })

# 4. INJECT FRAUD RING 3: "The Local Scammer" 
# Receives 18 payments (Just above threshold)
local_mule = "LOCAL_FRAUD@paytm"
for _ in range(18):
    data.append({
        "Transaction_ID": f"RING3_{random.randint(1000,9999)}",
        "Sender_ID": get_random_upi(),
        "Receiver_ID": local_mule,
        "Amount": 5000,
        "Timestamp": base_time + timedelta(hours=2),
        "Location": "Bangalore"
    })

# 5. INJECT IMPOSSIBLE TRAVEL (2 Different Thieves)
thieves = ["FAST_THIEF_1@ybl", "FAST_THIEF_2@okhdfc"]
for thief in thieves:
    # Transaction A (City 1)
    data.append({
        "Transaction_ID": f"JUMP_{thief}_1",
        "Sender_ID": thief, "Receiver_ID": "Shop_X@paytm",
        "Amount": 2000, "Timestamp": START_DATE, "Location": "Mumbai"
    })
    # Transaction B (City 2 - 10 mins later)
    data.append({
        "Transaction_ID": f"JUMP_{thief}_2",
        "Sender_ID": thief, "Receiver_ID": "Shop_Y@paytm",
        "Amount": 9000, "Timestamp": START_DATE + timedelta(minutes=10), "Location": "Delhi"
    })

# 6. SAVE
df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True) # Shuffle
df.to_csv("large_upi_dataset.csv", index=False)

print("Success! Created 3 Mule Rings and 2 Travelers.")