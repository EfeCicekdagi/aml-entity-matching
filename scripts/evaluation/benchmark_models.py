import os
import sys
import time
import yaml
import subprocess
import psycopg2
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

CONFIG_FILE = "config/aml_config.yaml"
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
        
def get_db_conn():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    db = config["database"]
    return psycopg2.connect(
        host=db["host"], port=db["port"],
        dbname=db["name"], user=db["user"], password=db["password"]
    )

def evaluate_results():
    """PostgreSQL'den en son run'ın sonuçlarını değerlendirir."""
    try:
        conn = get_db_conn()
        cur = conn.cursor()

        # En son başarılı run_id'yi bul
        cur.execute("""
            SELECT run_id FROM aml_run_log
            WHERE status = 'SUCCESS'
            ORDER BY finished_at DESC LIMIT 1
        """)
        row = cur.fetchone()
        if not row:
            conn.close()
            return 0, 0, 0, 0
        run_id = row[0]

        # Bu run'ın alertlerini çek
        cur.execute("""
            SELECT eft_id::TEXT, company_id, risk_level
            FROM aml_alert WHERE run_id = %s
        """, (run_id,))
        results = {}
        for eft_id, company_id, risk_level in cur.fetchall():
            if not eft_id.startswith("EFT_"):
                eft_id = f"EFT_{str(eft_id).zfill(5)}"
            results[eft_id] = (company_id, risk_level)

        # Ground truth tablosundan çek
        cur.execute("SELECT eft_id, true_company_id FROM aml_ground_truth")
        gt_rows = cur.fetchall()
        conn.close()

        if not gt_rows:
            return 0, 0, 0, 0

        y_true, y_pred = [], []
        exact_matches = 0
        total_known = 0

        for eft_id, true_company_id in gt_rows:
            is_match_true = 1 if true_company_id != -1 else 0
            if is_match_true:
                total_known += 1

            pred = results.get(eft_id, (None, "No Match"))
            pred_company_id, pred_risk = pred
            is_match_pred = 1 if (pred_company_id is not None and pred_risk not in ["No Match", "LOW"]) else 0

            y_true.append(is_match_true)
            y_pred.append(is_match_pred)

            if true_company_id != -1 and pred_company_id == true_company_id:
                exact_matches += 1

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        f1        = f1_score(y_true, y_pred, zero_division=0)
        exact_acc = exact_matches / total_known if total_known > 0 else 0

        return precision, recall, f1, exact_acc

    except Exception as e:
        print(f"Evaluation error: {e}")
        return 0, 0, 0, 0

def run_subprocess(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {cmd}:\n{result.stderr}")
    return result.returncode == 0

def benchmark():
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    
    existing_models = set()
    if os.path.exists(OUTPUT_MD):
        with open(OUTPUT_MD, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("| ") and not line.startswith("| Configuration |") and not line.startswith("|--------------|"):
                    model_name = line.split("|")[1].strip()
                    existing_models.add(model_name)
    else:
        with open(OUTPUT_MD, "w", encoding="utf-8") as f:
            f.write("# Model Benchmark Results\n\n")
            f.write("| Configuration | Precision | Recall | F1 Score | Exact Match Acc | Pipeline Time (s) |\n")
            f.write("|--------------|-----------|--------|----------|-----------------|-------------------|\n")

    current_embedding_model = None

    for cfg in MODELS_TO_TEST:
        if cfg['name'] in existing_models:
            print(f"Skipping {cfg['name']}, already benchmarked.")
            continue
            
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
