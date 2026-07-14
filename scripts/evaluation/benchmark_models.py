import os
import sys
import time
import yaml
import subprocess
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

CONFIG_FILE = "config/aml_config.yaml"
RESULTS_FILE = "outputs/best_matches.csv"
GROUND_TRUTH_FILE = "data/ground_truth.csv"
OUTPUT_MD = "outputs/benchmark_results.md"

MODELS_TO_TEST = [
    {
        "name": "bge-m3_ner-savasy_reranker-bge (Baseline)",
        "embedding": {"model_name": "BAAI/bge-m3", "dimension": 1024, "batch_size": 32},
        "ner": {"model_name": "savasy/bert-base-turkish-ner-cased"},
        "reranker": {"model_name": "BAAI/bge-reranker-v2-m3", "batch_size": 32}
    },
    {
        "name": "minilm_ner-savasy_reranker-bge",
        "embedding": {"model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "dimension": 384, "batch_size": 64},
        "ner": {"model_name": "savasy/bert-base-turkish-ner-cased"},
        "reranker": {"model_name": "BAAI/bge-reranker-v2-m3", "batch_size": 32}
    },
    {
        "name": "minilm_ner-dbmdz_reranker-mmarco",
        "embedding": {"model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "dimension": 384, "batch_size": 64},
        "ner": {"model_name": "dbmdz/bert-base-turkish-cased"},
        "reranker": {"model_name": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "batch_size": 32}
    },
    {
        "name": "e5-large_ner-savasy_reranker-bge",
        "embedding": {"model_name": "intfloat/multilingual-e5-large", "dimension": 1024, "batch_size": 32},
        "ner": {"model_name": "savasy/bert-base-turkish-ner-cased"},
        "reranker": {"model_name": "BAAI/bge-reranker-v2-m3", "batch_size": 32}
    }
]

def update_config(model_cfg):
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    config["embedding"]["model_name"] = model_cfg["embedding"]["model_name"]
    config["embedding"]["dimension"] = model_cfg["embedding"]["dimension"]
    config["embedding"]["batch_size"] = model_cfg["embedding"]["batch_size"]
    
    config["ner"]["model_name"] = model_cfg["ner"]["model_name"]
    
    config["reranker"]["model_name"] = model_cfg["reranker"]["model_name"]
    config["reranker"]["batch_size"] = model_cfg["reranker"]["batch_size"]
    
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
def evaluate_results():
    if not os.path.exists(RESULTS_FILE):
        return 0, 0, 0, 0

    results_df = pd.read_csv(RESULTS_FILE)
    gt_df = pd.read_csv(GROUND_TRUTH_FILE)
    
    results_df["eft_id"] = results_df["eft_id"].astype(str).apply(lambda x: x if x.startswith("EFT_") else f"EFT_{x.zfill(5)}")
    gt_df["eft_id"] = gt_df["eft_id"].astype(str)
    
    merged = pd.merge(gt_df, results_df, on="eft_id", how="left")
    merged["company_id"] = merged["company_id"].fillna(-1).astype(int)
    
    y_true, y_pred = [], []
    for _, row in merged.iterrows():
        is_known_true = 1 if row["true_company_id"] != -1 else 0
        is_known_pred = 1 if (row["company_id"] != -1 and row["risk_level"] != "No Match") else 0
        y_true.append(is_known_true)
        y_pred.append(is_known_pred)
        
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    exact_matches = sum(1 for _, row in merged.iterrows() if row["true_company_id"] != -1 and row["true_company_id"] == row["company_id"])
    total_known = sum(1 for _, row in merged.iterrows() if row["true_company_id"] != -1)
    exact_accuracy = exact_matches / total_known if total_known > 0 else 0
    
    return precision, recall, f1, exact_accuracy

def run_subprocess(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {cmd}:\n{result.stderr}")
    return result.returncode == 0

def benchmark():
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# Model Benchmark Results\n\n")
        f.write("| Configuration | Precision | Recall | F1 Score | Exact Match Acc | Pipeline Time (s) |\n")
        f.write("|--------------|-----------|--------|----------|-----------------|-------------------|\n")

    current_embedding_model = None

    for cfg in MODELS_TO_TEST:
        print(f"\n{'='*50}\nTesting configuration: {cfg['name']}\n{'='*50}")
        
        # 1. Update config
        update_config(cfg)
        
        # 2. Run ETL if embedding model changed
        emb_model = cfg["embedding"]["model_name"]
        if emb_model != current_embedding_model:
            print(f"Embedding model changed to {emb_model}. Running migration...")
            success = run_subprocess(f"{sys.executable} src/etl/migrate_embeddings.py")
            if not success:
                print("Migration failed. Skipping this configuration.")
                continue
            current_embedding_model = emb_model
            
        # 3. Clear reranker cache (optional, but good for fair time comparison)
        # We can just let it run. But we should measure time.
        
        # 4. Run main pipeline
        start_time = time.time()
        success = run_subprocess(f"{sys.executable} src/main.py")
        end_time = time.time()
        
        pipeline_time = end_time - start_time
        
        if not success:
            print("Pipeline failed. Skipping evaluation.")
            continue
            
        # 5. Evaluate
        precision, recall, f1, exact_acc = evaluate_results()
        
        # 6. Append to MD
        with open(OUTPUT_MD, "a", encoding="utf-8") as f:
            f.write(f"| {cfg['name']} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {exact_acc:.4f} | {pipeline_time:.2f} |\n")
            
        print(f"Results for {cfg['name']}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, Acc={exact_acc:.4f}, Time={pipeline_time:.2f}s")

    print("\nBenchmarking complete! See outputs/benchmark_results.md")

if __name__ == "__main__":
    benchmark()
