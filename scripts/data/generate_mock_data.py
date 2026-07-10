import pandas as pd
import random
import os

COMPANY_FILE = "data/company_list.csv"
MOCK_EFT_OUTPUT = "data/mock_bank_efts.csv"
GROUND_TRUTH_OUTPUT = "data/ground_truth.csv"

# Noise to inject
PREFIXES = ["TRF ", "INV ", "PAYMENT ", "SWIFT ", "HAVALE ", "EFT ", "TO ", "TRANSFER "]
SUFFIXES = [" ODEMESI", " INV", " TRF", " LLC", " LTD", " A.S.", " CORP"]

def add_noise(company_name):
    # 1. Randomly add prefix/suffix
    name = company_name
    if random.random() < 0.5:
        name = random.choice(PREFIXES) + name
    if random.random() < 0.5:
        name = name + random.choice(SUFFIXES)
    
    # 2. Randomly drop characters to simulate typos
    if random.random() < 0.3:
        idx = random.randint(0, len(name) - 1)
        name = name[:idx] + name[idx+1:]
        
    # 3. Randomly drop a word to simulate partial names
    words = name.split()
    if len(words) > 2 and random.random() < 0.2:
        words.pop(random.randint(0, len(words) - 1))
        name = " ".join(words)
        
    # 4. Uppercase / lowercase random
    if random.random() < 0.3:
        name = name.lower()
    elif random.random() < 0.3:
        name = name.upper()
        
    return name

def generate_mock_data(num_samples=1000):
    os.makedirs("data", exist_ok=True)
    df = pd.read_csv(COMPANY_FILE)
    
    companies = df.to_dict('records')
    
    mock_data = []
    ground_truth = []
    
    for i in range(1, num_samples + 1):
        # 80% chance it's a known company, 20% chance it's a random "No Match" company
        if random.random() < 0.8:
            company = random.choice(companies)
            description = add_noise(company["company_name"])
            true_company_id = company["company_id"]
        else:
            description = add_noise(random.choice(["Random Company", "Unknown Entity", "XYZ Holdings", "Nobody Inc", "Local Store"]))
            true_company_id = -1 # No match
            
        eft_id = f"EFT_{i:05d}"
        
        mock_data.append({
            "eft_id": eft_id,
            "description": description
        })
        
        ground_truth.append({
            "eft_id": eft_id,
            "true_company_id": true_company_id
        })
        
    mock_df = pd.DataFrame(mock_data)
    gt_df = pd.DataFrame(ground_truth)
    
    mock_df.to_csv(MOCK_EFT_OUTPUT, index=False)
    gt_df.to_csv(GROUND_TRUTH_OUTPUT, index=False)
    
    print(f"Generated {num_samples} mock EFTs.")
    print(f"Saved mock EFTs to {MOCK_EFT_OUTPUT}")
    print(f"Saved ground truth to {GROUND_TRUTH_OUTPUT}")

if __name__ == "__main__":
    generate_mock_data(1000)
