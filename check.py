import pandas as pd
from src.ui.dashboard import get_repo
def run():
    repo = get_repo()
    conn = repo.get_connection()
    df = pd.read_sql("SELECT * FROM aml_source.test_eft_input LIMIT 1", conn)
    print("test_eft_input columns:", df.columns.tolist())
    df2 = pd.read_sql("SELECT * FROM aml_eval.test_case LIMIT 1", conn)
    print("test_case columns:", df2.columns.tolist())
    repo.release_connection(conn)
run()
