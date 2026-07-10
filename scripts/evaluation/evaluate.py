import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

RESULTS_FILE = "outputs/best_matches.csv"
GROUND_TRUTH_FILE = "data/ground_truth.csv"

def evaluate():
    try:
        results_df = pd.read_csv(RESULTS_FILE)
        gt_df = pd.read_csv(GROUND_TRUTH_FILE)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Make sure you have run the main pipeline first.")
        return

    # Merge results with ground truth on eft_id
    merged = pd.merge(gt_df, results_df, on="eft_id", how="left")
    
    # Fill NaN values for best_matches (meaning no match was found)
    merged["company_id"] = merged["company_id"].fillna(-1).astype(int)
    
    y_true = []
    y_pred = []
    
    for _, row in merged.iterrows():
        # Binary classification: 1 if it's a known company, 0 if it's a random company
        is_known_true = 1 if row["true_company_id"] != -1 else 0
        
        # If model predicted a company and it's not 'No Match' risk level
        is_known_pred = 1 if (row["company_id"] != -1 and row["risk_level"] != "No Match") else 0
        
        y_true.append(is_known_true)
        y_pred.append(is_known_pred)
        
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate exact matches (how many times it found the EXACT right company)
    exact_matches = sum(1 for _, row in merged.iterrows() if row["true_company_id"] != -1 and row["true_company_id"] == row["company_id"])
    total_known = sum(1 for _, row in merged.iterrows() if row["true_company_id"] != -1)
    exact_accuracy = exact_matches / total_known if total_known > 0 else 0
    
    print("-" * 50)
    print("EVALUATION RESULTS")
    print("-" * 50)
    print(f"Total Evaluated EFTs : {len(merged)}")
    print(f"Total True Positives : {sum(y_true)} (Known Companies)")
    print("-" * 50)
    print(f"Detection Precision  : {precision:.4f} (When model says it's a known company, how often is it right?)")
    print(f"Detection Recall     : {recall:.4f} (Out of all known companies, how many did the model find?)")
    print(f"Detection F1 Score   : {f1:.4f}")
    print("-" * 50)
    print(f"Exact Match Accuracy : {exact_accuracy:.4f} (Did it pick the EXACT right company_id?)")
    print("-" * 50)

if __name__ == "__main__":
    evaluate()
