import psycopg2

conn = psycopg2.connect(host='127.0.0.1', port=5433, dbname='aml_db', user='postgres', password='password')
cur = conn.cursor()

cur.execute("""
    SELECT table_name FROM information_schema.tables 
    WHERE table_schema = 'public' 
    ORDER BY table_name
""")
tables = cur.fetchall()
print('=== TABLOLAR ===')
for t in tables:
    try:
        cur.execute(f'SELECT COUNT(*) FROM {t[0]}')
        count = cur.fetchone()[0]
        print(f'  {t[0]}: {count} satir')
    except Exception as e:
        print(f'  {t[0]}: HATA - {e}')
        conn.rollback()

# Son run'un alertleri
print('\n=== SON RUN ALERTLERI (aml_alert) ===')
cur.execute("""
    SELECT risk_level, COUNT(*) 
    FROM aml_alert 
    GROUP BY risk_level
    ORDER BY COUNT(*) DESC
""")
for row in cur.fetchall():
    print(f'  {row[0]}: {row[1]}')

print('\n=== RUN LOGLARI (Son 5 Run) ===')
cur.execute("""
    SELECT 
        run_id, 
        status, 
        duration_seconds,
        input_row_count, 
        prescreen_skipped_count,
        alert_count, 
        embedding_model,
        reranker_model,
        precision_score,
        recall_score,
        started_at 
    FROM aml_run_log 
    ORDER BY started_at DESC 
    LIMIT 5
""")
for row in cur.fetchall():
    r_id, stat, dur, inp, skip, alerts, emb, rer, prec, rec, start = row
    dur_str = f"{dur}s" if dur is not None else "?"
    skip_str = f"{skip}" if skip is not None else "?"
    prec_str = f"{prec:.2f}" if prec is not None else "?"
    rec_str = f"{rec:.2f}" if rec is not None else "?"
    
    print(f"  {r_id} | {stat} | {dur_str} | In:{inp} | Skip:{skip_str} | Alerts:{alerts} | Prec:{prec_str} | Rec:{rec_str}")
    if emb or rer:
        print(f"    -> Emb: {emb} | Rerank: {rer}")

conn.close()
