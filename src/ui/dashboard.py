"""
AML Alert Review Dashboard — Streamlit
Calistirilmak icin: streamlit run src/ui/dashboard.py
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.utils.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository
from src.utils.text_utils import normalize_text
from src.config.db_tables import TABLES
from src.etl.batch_processor import _acronym_score, _rule_score, _exact_name_score

# ── Sayfa ayarlari ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AML Alert Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
  .high-badge   { background:#ef4444; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.8rem; }
  .medium-badge { background:#f97316; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.8rem; }
  .low-badge    { background:#6b7280; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)

# ── DB Baglantisi ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_repo():
    cfg = ConfigLoader().get_db_config()
    return AMLRepository(host=cfg['host'], port=cfg['port'],
                         dbname=cfg['name'], user=cfg['user'], password=cfg['password'])

@st.cache_data(ttl=60)
def load_runs():
    repo = get_repo()
    conn = repo.get_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        df = pd.read_sql(f"""
            SELECT r.run_id, r.started_at, r.finished_at,
                   ROUND(EXTRACT(EPOCH FROM (r.finished_at-r.started_at))/60,1) AS sure_dk,
                   r.processed_row_count AS input_row_count, 
                   r.candidate_count, 
                   r.alert_count, r.status
            FROM {TABLES['run_log']} r ORDER BY r.started_at DESC LIMIT 20
        """, conn)
    finally:
        repo.release_connection(conn)
    return df

@st.cache_data(ttl=30)
def load_alerts(run_id):
    repo = get_repo()
    conn = repo.get_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        df = pd.read_sql(f"""
            SELECT a.alert_id, a.eft_id, 
                   COALESCE(e1.transaction_date, e2.transaction_date) as transaction_date, 
                   COALESCE(e1.amount, e2.amount) as amount, 
                   COALESCE(e1.sender_account_id, e2.sender_account_id) as sender_account_id, 
                   COALESCE(e1.receiver_account_id, e2.receiver_account_id) as receiver_account_id, 
                   COALESCE(e1.explanation, e2.explanation) AS original_explanation, 
                   COALESCE(e1.source_system, e2.source_system) as source_system, 
                   COALESCE(e1.batch_id, e2.batch_id) as batch_id,
                   v.original_company_name,
                   ROUND(a.final_score::numeric, 3) AS final_score,
                   ROUND(a.fuzzy_score::numeric, 3) AS fuzzy_score,
                   ROUND(a.vector_score::numeric, 3) AS vector_score,
                   ROUND(a.reranker_score::numeric, 3) AS reranker_score,
                   a.risk_level, a.alert_status, a.extracted_entity, a.created_at
            FROM {TABLES['alert']} a
            JOIN {TABLES['company_variant']} v ON a.variant_id=v.variant_id
            LEFT JOIN {TABLES['eft_input']} e1 ON a.eft_id = e1.eft_id
            LEFT JOIN aml_source.test_eft_input e2 ON a.eft_id = e2.eft_id
            WHERE a.run_id = %(run_id)s
            ORDER BY a.final_score DESC
        """, conn, params={"run_id": run_id})
    finally:
        repo.release_connection(conn)
    return df

@st.cache_resource
def load_live_components():
    import torch
    from sentence_transformers import SentenceTransformer
    from src.utils.ner_extractor import NERExtractor
    from src.retrieval.postgres_candidate_retriever import PostgresCandidateRetriever
    from src.reranker.reranker import Reranker
    from src.scoring.final_scorer import FinalScorer
    
    config_loader = ConfigLoader()
    config = config_loader.config
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    emb_model_name = config.get("embedding", {}).get("model_name", "BAAI/bge-m3")
    emb_model = SentenceTransformer(emb_model_name, device=device)
    
    ner_model = config.get("ner", {}).get("model_name", "savasy/bert-base-turkish-ner-cased")
    dev_id = 0 if device == "cuda" else -1
    ner_extractor = NERExtractor(model_name=ner_model, device=dev_id)
    
    repo = get_repo()
    retriever = PostgresCandidateRetriever(repo, config.get("retrieval", {}))
    reranker = Reranker(repo, config.get("reranker", {}))
    scorer = FinalScorer(
        repo, 
        config_version=config.get("scoring", {}).get("scoring_config_version", "scoring_v2_reranker"),
        threshold_version=config.get("scoring", {}).get("threshold_config_version", "threshold_v2_reranker")
    )
    
    return emb_model, ner_extractor, retriever, reranker, scorer


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚨 AML Dashboard")
    st.divider()
    
    page = st.radio("Menü", ["🏠 Ana Sayfa", "📈 Run Detayları"])
    st.divider()

    runs_df = load_runs()
    selected_run = None
    risk_filter = ["HIGH", "MEDIUM"]
    score_min = 0.50

    if page == "📈 Run Detayları":
        if runs_df.empty:
            st.error("Hiçbir run bulunamadı.")
            st.stop()

        run_options = runs_df['run_id'].tolist()
        selected_run = st.selectbox("Run Seç", run_options,
                                     format_func=lambda x: f"{x}  ({runs_df[runs_df.run_id==x]['started_at'].values[0]})")

        st.divider()
        # Tum DB verisi sayisi
        try:
            conn = get_repo().get_connection()
            total_db_rows = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {TABLES['eft_input']}", conn).iloc[0]['cnt']
        except Exception:
            total_db_rows = 0
        finally:
            if conn: get_repo().release_connection(conn)
            
        st.markdown(f"**Veritabanındaki Toplam Veri:** {total_db_rows:,} EFT")
        
        run_info = runs_df[runs_df['run_id'] == selected_run].iloc[0]
        st.markdown(f"**İşlenen Girdi (Run):** {int(run_info['input_row_count']):,}")
        st.markdown(f"**2. Aşamaya Geçen (Aday):** {int(run_info['candidate_count']):,}")
        st.markdown(f"**Süre:** {run_info['sure_dk']} dk")
        st.markdown(f"**Durum:** {'✅' if run_info['status']=='SUCCESS' else '❌'} {run_info['status']}")

        st.divider()
        risk_filter = st.multiselect("Risk Filtresi", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"])
        score_min = st.slider("Min Final Score", 0.0, 1.0, 0.50, 0.01)
        st.divider()

    if st.button("🔄 Yenile"):
        st.cache_data.clear()
        st.rerun()

# ── Ana icerik ────────────────────────────────────────────────────────────────

if page == "🏠 Ana Sayfa":
    st.title("AML Entity Matching Projesi")
    st.write("Sisteme hoş geldiniz! Aşağıdaki grafikten genel durumu inceleyebilir veya Canlı Test bölümünden modelleri test edebilirsiniz.")
    st.divider()

    st.markdown("### 📊 Tüm Run'larda Alert Trendi")
    if not runs_df.empty:
        fig3 = px.bar(runs_df.sort_values('started_at'), x='run_id', y='alert_count',
                      text='alert_count', color='alert_count',
                      color_continuous_scale='Reds')
        fig3.update_traces(textposition='outside')
        fig3.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Kayıtlı run bulunmamaktadır.")

    st.divider()

    st.markdown("### ⚡ Canlı AML Testi")
    st.write("EFT açıklamasını veya şirket adını girin. Sistem anlık olarak analiz edip eşleşmeleri bulacaktır.")
    
    user_input = st.text_area("Test Metni (EFT Açıklaması / İsim)", height=100)
    
    if st.button("🚀 Test Et", type="primary"):
        if not user_input.strip():
            st.warning("Lütfen test etmek için bir metin girin.")
        else:
            with st.spinner("Modeller yükleniyor ve metin analiz ediliyor..."):
                emb_model, ner_extractor, retriever, reranker, scorer = load_live_components()
                
                # 1. Normalization & NER
                norm_exp = normalize_text(user_input).lower()
                entity = ner_extractor.extract_entity(user_input)
                
                st.markdown(f"**Çıkarılan Varlık (NER):** `{entity if entity else 'Bulunamadı'}`")
                
                # 2. Embedding
                embedding = emb_model.encode([norm_exp], show_progress_bar=False)[0]
                
                # 3. Retrieval
                row_data = [{
                    "row_id": "live_test",
                    "normalized_explanation": norm_exp,
                    "embedding": embedding.tolist(),
                    "extracted_entity": entity
                }]
                candidates_dict = retriever.batch_get_candidates(row_data)
                candidates = candidates_dict.get("live_test", [])
                
                if not candidates:
                    st.success("✅ Veritabanındaki yasaklı/şüpheli listesinde eşleşme bulunamadı. (Sıfır risk)")
                else:
                    # 4. Reranking
                    strong = reranker.score_candidates(norm_exp, candidates)
                    
                    # 5. Scoring
                    results = []
                    for cand in strong:
                        fuzzy_score  = cand["candidate_score"] if "pg_trgm" in cand.get("sources", []) else 0.0
                        vector_score = cand["candidate_score"] if "pgvector" in cand.get("sources", []) else 0.0
                        scores_dict = {
                            "fuzzy_score":    fuzzy_score,
                            "vector_score":   vector_score,
                            "acronym_score":  _acronym_score(norm_exp, cand["variant_name"]),
                            "rule_score":     max(
                                _rule_score(norm_exp, cand["variant_name"]),
                                _exact_name_score(norm_exp, cand["variant_name"])
                            ),
                            "reranker_score": cand.get("reranker_score", 0.0),
                        }
                        
                        if entity:
                            import difflib
                            fuzzy_ext = difflib.SequenceMatcher(None, entity.lower(), cand["variant_name"].lower()).ratio()
                            scores_dict["fuzzy_score"] = max(scores_dict["fuzzy_score"], fuzzy_ext)
                            
                            acronym_ext = _acronym_score(entity, cand["variant_name"])
                            scores_dict["acronym_score"] = max(scores_dict["acronym_score"], acronym_ext)
                            
                            rule_ext = max(_rule_score(entity, cand["variant_name"]), _exact_name_score(entity, cand["variant_name"]))
                            scores_dict["rule_score"] = max(scores_dict["rule_score"], rule_ext)
                            
                        final_score = scorer.calculate_final_score(scores_dict)
                        risk_level  = scorer.assign_risk_level(final_score)
                        
                        results.append({
                            "Şirket": cand["company_name"],
                            "Varyant": cand["variant_name"],
                            "Risk": risk_level,
                            "Final Skor": final_score,
                            "Reranker Skor": scores_dict["reranker_score"],
                            "Vektör Skor": scores_dict["vector_score"],
                            "Fuzzy Skor": scores_dict["fuzzy_score"]
                        })
                    
                    if not results:
                        st.success("✅ Eşleşme bulundu ancak skorlar risk oluşturacak düzeyde değil.")
                    else:
                        res_df = pd.DataFrame(results).sort_values("Final Skor", ascending=False)
                        
                        # Risk badge fonksiyonu
                        def format_risk(risk):
                            if risk == "HIGH": return "🔴 HIGH"
                            if risk == "MEDIUM": return "🟠 MEDIUM"
                            return f"⚪ {risk}"
                            
                        res_df["Risk"] = res_df["Risk"].apply(format_risk)
                        
                        st.markdown("#### Bulunan Eşleşmeler")
                        st.dataframe(res_df, use_container_width=True, column_config={
                            "Final Skor": st.column_config.ProgressColumn(
                                "Final Skor", min_value=0, max_value=1, format="%.3f"
                            )
                        })


elif page == "📈 Run Detayları":
    st.title(f"🚨 AML Alert Review — {selected_run}")

    alerts_df = load_alerts(selected_run)
    if alerts_df.empty:
        st.warning("Bu run icin alert bulunamadi.")

    # Filtrele
    filtered = alerts_df[
        (alerts_df['risk_level'].isin(risk_filter)) &
        (alerts_df['final_score'] >= score_min)
    ]

    # ── KPI Metrikleri ────────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Toplam Alert", f"{len(alerts_df):,}")
    with col2:
        high_cnt = len(alerts_df[alerts_df['risk_level']=='HIGH']) if not alerts_df.empty else 0
        st.metric("🔴 HIGH", f"{high_cnt:,}")
    with col3:
        med_cnt = len(alerts_df[alerts_df['risk_level']=='MEDIUM']) if not alerts_df.empty else 0
        st.metric("🟠 MEDIUM", f"{med_cnt:,}")
    with col4:
        avg_score = alerts_df['final_score'].mean() if not alerts_df.empty else 0.0
        st.metric("Ortalama Skor", f"{avg_score:.3f}")
    with col5:
        st.metric("Gösterilen", f"{len(filtered):,}")

    st.divider()

    # ── Grafikler ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Dağılım Analizi", "📋 Alert Listesi", "🏢 Şirket Analizi"])

    with tab1:
        if alerts_df.empty:
            st.info("Bu run için gösterilecek grafik verisi bulunamadı.")
        else:
            col_l, col_r = st.columns(2)

            with col_l:
                # Score histogram
                fig = px.histogram(alerts_df, x="final_score", nbins=40,
                                   color="risk_level",
                                   color_discrete_map={"HIGH":"#ef4444","MEDIUM":"#f97316","LOW":"#6b7280"},
                                   title="Final Score Dağılımı",
                                   labels={"final_score":"Final Score","count":"Alert Sayısı"})
                fig.add_vline(x=0.70, line_dash="dash", line_color="#ef4444",
                              annotation_text="HIGH eşiği (0.70)")
                fig.add_vline(x=0.62, line_dash="dash", line_color="#f97316",
                              annotation_text="MEDIUM eşiği (0.62)")
                fig.update_layout(bargap=0.05, height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col_r:
                # Risk pie chart
                risk_counts = alerts_df['risk_level'].value_counts().reset_index()
                risk_counts.columns = ['risk_level','count']
                fig2 = px.pie(risk_counts, values='count', names='risk_level',
                              title="Risk Seviyesi Dağılımı",
                              color='risk_level',
                              color_discrete_map={"HIGH":"#ef4444","MEDIUM":"#f97316","LOW":"#6b7280"})
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        if filtered.empty:
            st.info("Bu run için gösterilecek alert listesi bulunamadı.")
        else:
            st.markdown(f"**{len(filtered):,} alert gösteriliyor** (filtre: risk={risk_filter}, score≥{score_min})")

            # Risk badge fonksiyonu
            def badge(risk):
                colors = {"HIGH":"🔴","MEDIUM":"🟠","LOW":"⚪"}
                return f"{colors.get(risk,'')} {risk}"

            display_df = filtered[['eft_id', 'transaction_date', 'amount', 'sender_account_id', 'receiver_account_id', 'original_explanation', 'source_system', 'batch_id', 'original_company_name','extracted_entity','final_score','reranker_score','vector_score','fuzzy_score','risk_level','alert_status','created_at']].copy()
            display_df['risk_level'] = display_df['risk_level'].apply(badge)
            display_df.columns = ['EFT ID', 'İşlem Tarihi', 'Tutar', 'Gönderen Hesap', 'Alıcı Hesap', 'EFT Açıklaması', 'Kaynak Sistem', 'Batch ID', 'Benzediği Şirket', 'Çıkarılan İsim (NER)', 'Final Skor', 'Reranker Skoru', 'Vektör Skoru', 'Fuzzy Skoru', 'Tehlike Kategorisi (Risk)', 'Durum', 'Tarih']

            st.dataframe(display_df, use_container_width=True, height=500,
                         column_config={
                             "Final Skor": st.column_config.ProgressColumn(
                                 "Final Skor", min_value=0, max_value=1, format="%.3f"
                             )
                         })

            # CSV indirme
            csv = filtered.to_csv(index=False).encode('utf-8')
            st.download_button("📥 CSV İndir", csv, f"alerts_{selected_run}.csv", "text/csv")

    with tab3:
        if alerts_df.empty:
            st.info("Şirket analizi yapılabilecek bir veri yok.")
        else:
            company_agg = alerts_df.groupby('original_company_name').agg(
                alert_sayisi=('alert_id','count'),
                avg_skor=('final_score','mean'),
                high_cnt=('risk_level', lambda x: (x=='HIGH').sum()),
                medium_cnt=('risk_level', lambda x: (x=='MEDIUM').sum()),
            ).reset_index().sort_values('alert_sayisi', ascending=False)
            company_agg['avg_skor'] = company_agg['avg_skor'].round(3)

            fig4 = px.bar(company_agg, x='original_company_name', y='alert_sayisi',
                          color='avg_skor', color_continuous_scale='Reds',
                          title="Şirket Bazında Alert Sayısı",
                          text='alert_sayisi')
            fig4.update_traces(textposition='outside')
            fig4.update_layout(height=400, xaxis_tickangle=-30)
            st.plotly_chart(fig4, use_container_width=True)

            st.dataframe(company_agg.rename(columns={
                'original_company_name':'Şirket',
                'alert_sayisi':'Alert',
                'avg_skor':'Avg Score',
                'high_cnt':'HIGH',
                'medium_cnt':'MEDIUM'
            }), use_container_width=True)

