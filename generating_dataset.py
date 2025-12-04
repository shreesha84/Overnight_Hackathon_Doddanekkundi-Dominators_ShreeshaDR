import pandas as pd
import random
import uuid
from datetime import datetime, timedelta
import numpy as np

# --- CONFIGURATION ---
NUM_ROWS = 8000  # Base legitimate transactions
START_DATE = datetime(2024, 1, 1, 8, 0, 0)
END_DATE = datetime(2024, 1, 7, 23, 59, 59)

# --- HELPER LISTS ---
banks = ["@oksbi", "@okhdfc", "@ybl", "@paytm", "@ibl", "@axl", "@icici"]
first_names = ["arjun", "priya", "rahul", "sneha", "vikram", "amit", "pooja", "karan", 
               "simran", "rohan", "ananya", "dev", "isha", "aditya", "neha"]
locations = ["Mumbai", "Delhi", "Bangalore", "Pune", "Chennai", "Kolkata", 
             "Hyderabad", "Ahmedabad", "Jaipur", "Lucknow"]
merchant_types = ["food", "transport", "bills", "shopping", "entertainment", "health", "education"]
merchants = {
    "food": ["Swiggy", "Zomato", "DelhiBelly", "CafeShop", "FoodCourt"],
    "transport": ["Uber", "Ola", "MetroCard", "PetrolPump", "Rapido"],
    "bills": ["Electricity", "Water", "Gas", "Internet", "Mobile"],
    "shopping": ["Amazon", "Flipkart", "LocalStore", "DMart", "Reliance"],
    "entertainment": ["BookMyShow", "Netflix", "Hotstar", "Gaming", "Spotify"],
    "health": ["Pharmacy", "Hospital", "Clinic", "HealthApp"],
    "education": ["CourseApp", "School", "Tuition", "Books"]
}

def get_random_upi(name_pool=first_names):
    return f"{random.choice(name_pool)}_{random.randint(100, 9999)}{random.choice(banks)}"

def get_merchant_upi(category):
    merchant = random.choice(merchants[category])
    return f"{merchant.lower()}.pay{random.choice(banks)}"

def get_random_timestamp(start, end, time_pattern="normal"):
    """Generate realistic timestamps based on patterns"""
    total_seconds = int((end - start).total_seconds())
    random_second = random.randint(0, total_seconds)
    base_time = start + timedelta(seconds=random_second)
    
    if time_pattern == "normal":
        # Weight towards daytime (8AM-10PM)
        hour = random.choices(range(24), weights=[1,1,2,3,4,8,12,15,18,20,20,18,16,14,12,10,8,12,16,18,12,8,4,2])[0]
        base_time = base_time.replace(hour=hour)
    elif time_pattern == "night":
        # Suspicious night hours (11PM-4AM)
        hour = random.choice([23, 0, 1, 2, 3, 4])
        base_time = base_time.replace(hour=hour)
    
    return base_time

print(f"Generating comprehensive UPI fraud detection dataset...")
data = []

# ==============================================
# 1. GENERATE REALISTIC LEGITIMATE TRANSACTIONS
# ==============================================
print("Generating legitimate transactions...")

# Create a pool of regular users
regular_users = [get_random_upi() for _ in range(500)]
user_profiles = {}

for user in regular_users:
    user_profiles[user] = {
        "home_location": random.choice(locations),
        "avg_transaction": random.randint(200, 2000),
        "preferred_merchants": random.sample(merchant_types, k=3)
    }

for _ in range(NUM_ROWS):
    sender = random.choice(regular_users)
    profile = user_profiles[sender]
    
    # Choose transaction type
    if random.random() < 0.6:  # 60% merchant payments
        category = random.choice(profile["preferred_merchants"])
        receiver = get_merchant_upi(category)
        merchant_type = category
        
        # Realistic amounts per category
        if category == "food":
            amount = round(random.uniform(50, 800), 2)
        elif category == "transport":
            amount = round(random.uniform(30, 500), 2)
        elif category == "bills":
            amount = round(random.uniform(200, 5000), 2)
        elif category == "shopping":
            amount = round(random.uniform(100, 10000), 2)
        else:
            amount = round(random.uniform(100, 2000), 2)
    else:  # 40% P2P transfers
        receiver = random.choice(regular_users)
        merchant_type = "p2p"
        amount = round(random.gauss(profile["avg_transaction"], 500), 2)
        amount = max(10, min(amount, 25000))  # Cap amounts
    
    # Mostly home location, sometimes travel
    location = profile["home_location"] if random.random() < 0.85 else random.choice(locations)
    
    data.append({
        "Transaction_ID": str(uuid.uuid4())[:12],
        "Sender_ID": sender,
        "Receiver_ID": receiver,
        "Amount": amount,
        "Timestamp": get_random_timestamp(START_DATE, END_DATE, "normal"),
        "Location": location,
        "Merchant_Type": merchant_type,
        "Status": "success"
    })

# ==============================================
# 2. FRAUD PATTERN 1: MONEY MULE NETWORK
# ==============================================
print("Injecting Money Mule Networks...")

# Large mule network (kingpin)
kingpin_mule = "KINGPIN_BOSS@ybl"
kingpin_time = START_DATE + timedelta(hours=14)

for i in range(60):
    victim = get_random_upi()
    data.append({
        "Transaction_ID": f"MULE1_{uuid.uuid4().hex[:8]}",
        "Sender_ID": victim,
        "Receiver_ID": kingpin_mule,
        "Amount": round(random.uniform(500, 2000), 2),
        "Timestamp": kingpin_time + timedelta(minutes=random.randint(0, 120)),
        "Location": random.choice(["Mumbai", "Delhi", "Pune"]),
        "Merchant_Type": "p2p",
        "Status": "success"
    })

# Medium mule (phishing operator)
phishing_mule = "PHISHING_NET@oksbi"
for i in range(35):
    data.append({
        "Transaction_ID": f"MULE2_{uuid.uuid4().hex[:8]}",
        "Sender_ID": get_random_upi(),
        "Receiver_ID": phishing_mule,
        "Amount": round(random.uniform(100, 500), 2),  # Small amounts
        "Timestamp": kingpin_time + timedelta(hours=1, minutes=random.randint(0, 90)),
        "Location": "Delhi",
        "Merchant_Type": "p2p",
        "Status": "success"
    })

# Smaller mule (local scammer)
local_mule = "LOCAL_SCAM@paytm"
for i in range(22):
    data.append({
        "Transaction_ID": f"MULE3_{uuid.uuid4().hex[:8]}",
        "Sender_ID": get_random_upi(),
        "Receiver_ID": local_mule,
        "Amount": round(random.uniform(2000, 8000), 2),
        "Timestamp": kingpin_time + timedelta(hours=3, minutes=random.randint(0, 60)),
        "Location": "Bangalore",
        "Merchant_Type": "p2p",
        "Status": "success"
    })

# ==============================================
# 3. FRAUD PATTERN 2: IMPOSSIBLE TRAVEL
# ==============================================
print("Injecting Impossible Travel patterns...")

for i in range(5):
    thief = f"TRAVEL_THIEF_{i}@ybl"
    base_time = START_DATE + timedelta(hours=random.randint(10, 20))
    
    # Transaction in City A
    data.append({
        "Transaction_ID": f"TRAVEL_{i}_A",
        "Sender_ID": thief,
        "Receiver_ID": get_merchant_upi("shopping"),
        "Amount": round(random.uniform(1000, 5000), 2),
        "Timestamp": base_time,
        "Location": "Mumbai",
        "Merchant_Type": "shopping",
        "Status": "success"
    })
    
    # Transaction in City B (impossible - 10 mins later)
    data.append({
        "Transaction_ID": f"TRAVEL_{i}_B",
        "Sender_ID": thief,
        "Receiver_ID": get_merchant_upi("shopping"),
        "Amount": round(random.uniform(3000, 15000), 2),
        "Timestamp": base_time + timedelta(minutes=8),
        "Location": "Delhi",
        "Merchant_Type": "shopping",
        "Status": "success"
    })

# ==============================================
# 4. FRAUD PATTERN 3: VELOCITY ATTACKS
# ==============================================
print("Injecting High Velocity attacks...")

for i in range(3):
    velocity_attacker = f"VELOCITY_BOT_{i}@paytm"
    attack_time = START_DATE + timedelta(days=random.randint(1, 5), hours=random.randint(1, 20))
    
    # Burst of 15 transactions in 30 minutes
    for j in range(15):
        data.append({
            "Transaction_ID": f"VEL_{i}_{j}",
            "Sender_ID": velocity_attacker,
            "Receiver_ID": get_merchant_upi(random.choice(merchant_types)),
            "Amount": round(random.uniform(100, 1000), 2),
            "Timestamp": attack_time + timedelta(minutes=j*2),
            "Location": random.choice(locations),
            "Merchant_Type": random.choice(merchant_types),
            "Status": random.choice(["success", "failed"])
        })

# ==============================================
# 5. FRAUD PATTERN 4: ROUND AMOUNT TESTING
# ==============================================
print("Injecting Round Amount testing patterns...")

for i in range(4):
    tester = f"ROUND_TESTER_{i}@ybl"
    test_time = START_DATE + timedelta(days=random.randint(1, 6))
    
    # Small test transactions (round amounts)
    for amount in [10, 50, 100, 500, 1000]:
        data.append({
            "Transaction_ID": f"ROUND_{i}_{amount}",
            "Sender_ID": tester,
            "Receiver_ID": get_random_upi(),
            "Amount": float(amount),
            "Timestamp": test_time + timedelta(minutes=random.randint(1, 10)),
            "Location": random.choice(locations),
            "Merchant_Type": "p2p",
            "Status": "success"
        })
    
    # Followed by large round amount
    data.append({
        "Transaction_ID": f"ROUND_{i}_FINAL",
        "Sender_ID": tester,
        "Receiver_ID": get_random_upi(),
        "Amount": float(random.choice([5000, 10000, 15000])),
        "Timestamp": test_time + timedelta(minutes=30),
        "Location": random.choice(locations),
        "Merchant_Type": "p2p",
        "Status": "success"
    })

# ==============================================
# 6. FRAUD PATTERN 5: NIGHT TIME LARGE TRANSACTIONS
# ==============================================
print("Injecting Unusual Hour patterns...")

for i in range(8):
    data.append({
        "Transaction_ID": f"NIGHT_{i}",
        "Sender_ID": get_random_upi(),
        "Receiver_ID": get_random_upi(),
        "Amount": round(random.uniform(5000, 25000), 2),
        "Timestamp": get_random_timestamp(START_DATE, END_DATE, "night"),
        "Location": random.choice(locations),
        "Merchant_Type": "p2p",
        "Status": "success"
    })

# ==============================================
# 7. SAVE DATASET
# ==============================================
df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
df = df.sort_values('Timestamp').reset_index(drop=True)  # Sort by time
df.to_csv("upi_fraud_dataset.csv", index=False)

print(f"\n{'='*60}")
print(f"SUCCESS! Generated {len(df):,} transactions")
print(f"{'='*60}")
print(f"✓ {len(regular_users)} regular users with realistic behavior")
print(f"✓ 3 Money Mule networks (117 suspicious transactions)")
print(f"✓ 5 Impossible travel cases (10 transactions)")
print(f"✓ 3 High velocity attacks (45 transactions)")
print(f"✓ 4 Round amount testing patterns (24 transactions)")
print(f"✓ 8 Unusual hour large transactions")
print(f"\nFile saved: upi_fraud_dataset.csv")