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
from src.utils.text_utils import normalize_text, get_normalized_core_name, is_consonant_match
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
def load_css(file_path):
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        pass

css_path = os.path.join(os.path.dirname(__file__), "style.css")
load_css(css_path)

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
                   r.alert_count, r.status,
                   r.scoring_config_version, r.threshold_version, r.pipeline_version,
                   r.embedding_model_version, r.reranker_model_version,
                   COALESCE(r.high_alert_count, 0) AS high_alert_count,
                   COALESCE(r.medium_alert_count, 0) AS medium_alert_count,
                   COALESCE(r.no_candidate_count, 0) AS no_candidate_count,
                   COALESCE(r.match_result_count, 0) AS match_result_count,
                   COALESCE(r.p50_latency_ms, 0) AS p50_latency_ms,
                   COALESCE(r.p95_latency_ms, 0) AS p95_latency_ms,
                   COALESCE(r.p99_latency_ms, 0) AS p99_latency_ms,
                   r.watchlist_version, r.calibration_version,
                   r.embedding_model_hash, r.reranker_model_hash
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
                   a.risk_level, a.alert_status, a.extracted_entity, a.match_reason, a.created_at,
                   a.entity_extraction_status, a.matched_variant_name, a.variant_type,
                   a.watchlist_company_name, a.reviewed_by, a.reviewed_at, a.review_result,
                   a.analyst_note, a.false_positive_reason, a.status_updated_at,
                   COALESCE(a.decision_status, a.risk_level) AS decision_status,
                   a.reason_codes,
                   ROUND(COALESCE(a.calibrated_probability, a.reranker_score)::numeric, 3) AS calibrated_probability,
                   COALESCE(a.calibration_applied, false) AS calibration_applied,
                   a.entity_type, a.extraction_method,
                   COALESCE(a.candidate_count, 0) AS candidate_count,
                   a.human_explanation,
                   a.retrieval_sources
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
    
    page = st.radio("Menü", ["🏠 Ana Sayfa", "📈 Run Detayları", "🧪 Benchmark"])
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
        st.markdown(f"**Değerlendirilen Şirket Adayı:** {int(run_info['candidate_count']):,} (EFT başına N adet)")
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
    st.title("🚨 AML AI - Akıllı Varlık Eşleştirme Motoru")
    st.markdown("Sisteme hoş geldiniz! Bu panel üzerinden **EFT açıklamalarındaki şüpheli varlıkları** anlık olarak tespit edebilir ve arka planda çalışan yapay zeka modellerini test edebilirsiniz.")
    st.divider()

    st.markdown("### ⚡ Canlı AML Testi")
    st.write("EFT açıklamasını veya şüpheli şirket adını girin. Sistem saniyeler içinde veritabanındaki riskli varlıklarla benzerlik analizini yapacaktır.")
    
    user_input = st.text_area("Test Metni (EFT Açıklaması / İsim)", height=100, placeholder="Örn: TRANSFER FROM APPLE INC FOR LAPTOPS")
    
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
                candidates_res = retriever.batch_get_candidates(row_data).get("live_test")
                candidates = candidates_res.get("candidates", []) if candidates_res else []

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
                        norm_cand = normalize_text(cand["variant_name"])
                        core_query = get_normalized_core_name(norm_exp)
                        core_cand = get_normalized_core_name(cand["variant_name"])
                        
                        query_token_count = len(norm_exp.split())
                        
                        exact_normalized_match = (norm_exp == norm_cand and bool(norm_exp))
                        exact_core_match = (core_query == core_cand and bool(core_query))
                        legal_suffix_only_difference = exact_core_match and not exact_normalized_match
                        
                        query_is_contained = (norm_exp in norm_cand and bool(norm_exp))
                        cand_is_contained = (norm_cand in norm_exp and bool(norm_cand))

                        scores_dict = {
                            "fuzzy_score":    fuzzy_score,
                            "vector_score":   vector_score,
                            "acronym_score":  _acronym_score(norm_exp, cand["variant_name"]),
                            "rule_score":     max(
                                _rule_score(norm_exp, cand["variant_name"]),
                                _exact_name_score(norm_exp, cand["variant_name"])
                            ),
                            "reranker_score": cand.get("reranker_score", 0.0),
                            "query_token_count": query_token_count,
                            "exact_normalized_match": exact_normalized_match,
                            "exact_core_match": exact_core_match,
                            "legal_suffix_only_difference": legal_suffix_only_difference,
                            "query_is_contained_in_candidate": query_is_contained,
                            "candidate_is_contained_in_query": cand_is_contained,
                            "consonant_match": is_consonant_match(core_query, core_cand)
                        }
                        
                        if entity:
                            import difflib
                            fuzzy_ext = difflib.SequenceMatcher(None, entity.lower(), cand["variant_name"].lower()).ratio()
                            if core_cand:
                                fuzzy_ext_core = difflib.SequenceMatcher(None, entity.lower(), core_cand.lower()).ratio()
                                fuzzy_ext = max(fuzzy_ext, fuzzy_ext_core)
                            scores_dict["fuzzy_score"] = max(scores_dict["fuzzy_score"], fuzzy_ext)
                            
                            if is_consonant_match(entity, core_cand):
                                scores_dict["consonant_match"] = True
                            
                            acronym_ext = _acronym_score(entity, cand["variant_name"])
                            scores_dict["acronym_score"] = max(scores_dict["acronym_score"], acronym_ext)
                            
                            rule_ext = max(_rule_score(entity, cand["variant_name"]), _exact_name_score(entity, cand["variant_name"]))
                            scores_dict["rule_score"] = max(scores_dict["rule_score"], rule_ext)
                            
                        final_score, match_reason, reason_codes = scorer.calculate_final_score(scores_dict)
                        risk_level  = scorer.assign_risk_level(final_score)
                        
                        results.append({
                            "Şirket": cand["company_name"],
                            "Varyant": cand["variant_name"],
                            "Risk": risk_level,
                            "Sebep": match_reason,
                            "Gerekçeler": ", ".join(reason_codes),
                            "Final Skor": final_score,
                            "Reranker Skor": scores_dict["reranker_score"]
,
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

    st.divider()
    with st.expander("📊 Run Bazında Alert Sayısı", expanded=False):
        if not runs_df.empty:
            fig3 = px.bar(runs_df.sort_values('started_at'), x='run_id', y='alert_count',
                          text='alert_count', color='alert_count',
                          color_continuous_scale='Reds',
                          title="Run Başına Üretilen Alert Sayısı")
            fig3.update_traces(textposition='outside')
            fig3.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Kayıtlı run bulunmamaktadır.")


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
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
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
        run_row = runs_df[runs_df['run_id'] == selected_run].iloc[0]
        no_cand = int(run_row.get('no_candidate_count', 0)) if not runs_df.empty else 0
        st.metric("⚫ No Candidate", f"{no_cand:,}")
    with col6:
        p95 = run_row.get('p95_latency_ms', None) if not runs_df.empty else None
        st.metric("P95 Latency", f"{p95:.0f} ms" if p95 and p95 > 0 else "N/A")
    with col7:
        st.metric("Gösterilen", f"{len(filtered):,}")

    st.divider()
    
    # ── Run Configuration (Versiyonlar) ──────────────────────────────────────────
    run_info = runs_df[runs_df['run_id'] == selected_run].iloc[0]
    with st.expander("⚙️ Run Konfigürasyon Bilgileri", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**Pipeline:** `{run_info.get('pipeline_version', 'N/A')}`")
            st.markdown(f"**Embedding Model:** `{run_info.get('embedding_model_version', 'N/A')}`")
        with c2:
            st.markdown(f"**Scoring Config:** `{run_info.get('scoring_config_version', 'N/A')}`")
            st.markdown(f"**Reranker Model:** `{run_info.get('reranker_model_version', 'N/A')}`")
        with c3:
            st.markdown(f"**Threshold Config:** `{run_info.get('threshold_version', 'N/A')}`")

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

            display_df = filtered[['alert_id', 'eft_id', 'transaction_date', 'amount', 'original_explanation', 'original_company_name','matched_variant_name','variant_type','extracted_entity','entity_extraction_status','final_score','reranker_score','vector_score','fuzzy_score','risk_level','alert_status','reviewed_by','created_at']].copy()
            display_df['risk_level'] = display_df['risk_level'].apply(badge)
            display_df.columns = ['Alert ID', 'EFT ID', 'İşlem Tarihi', 'Tutar', 'EFT Açıklaması', 'Benzediği Şirket (Ana)', 'Eşleşen Varyant', 'Varyant Tipi', 'Çıkarılan Entity', 'Entity Extraction Durumu', 'Final Skor', 'Reranker Skoru', 'Vektör Skoru', 'Fuzzy Skoru', 'Tehlike Kategorisi (Risk)', 'Durum', 'İnceleyen', 'Tarih']

            st.dataframe(display_df, use_container_width=True, height=400,
                         column_config={
                             "Final Skor": st.column_config.ProgressColumn(
                                 "Final Skor", min_value=0, max_value=1, format="%.3f"
                             )
                         })

            # CSV indirme
            csv = filtered.to_csv(index=False).encode('utf-8')
            st.download_button("📥 CSV İndir", csv, f"alerts_{selected_run}.csv", "text/csv")
            
            st.markdown("---")
            st.subheader("🕵️‍♂️ Alert İnceleme ve Yönetimi")
            alert_ids = filtered['alert_id'].tolist()
            if alert_ids:
                selected_alert_id = st.selectbox("İncelenecek Alert ID", options=alert_ids)
                selected_alert = filtered[filtered['alert_id'] == selected_alert_id].iloc[0]
                
                with st.expander(f"Alert Detayları - {selected_alert_id}", expanded=True):
                    st.write(f"**EFT Açıklaması:** {selected_alert['original_explanation']}")
                    st.write(f"**Çıkarılan Entity:** {selected_alert.get('extracted_entity', '-')} ({selected_alert.get('entity_extraction_status', '-')})")
                    st.write(f"**Entity Türü:** {selected_alert.get('entity_type', '-')} | Extraction Yöntemi: {selected_alert.get('extraction_method', '-')}")
                    st.write(f"**Eşleşen Varyant:** {selected_alert.get('matched_variant_name', '-')} (Tip: {selected_alert.get('variant_type', '-')}) -> Ana Şirket: {selected_alert.get('original_company_name', '-')}")
                    st.write(f"**Match Reason:** {selected_alert.get('match_reason', '-')}")
                    st.write(f"**Decision Status:** {selected_alert.get('decision_status', '-')}")
                    # Kalibrasyon bilgisi
                    cal_prob = selected_alert.get('calibrated_probability')
                    cal_applied = selected_alert.get('calibration_applied', False)
                    if cal_prob is not None:
                        cal_label = '✅ Kalibre Edildi' if cal_applied else '⚠️ Ham Skor'
                        st.write(f"**Kalibre Olasılık:** {float(cal_prob):.3f} ({cal_label})")
                    # Reason codes
                    reason_codes = selected_alert.get('reason_codes')
                    if reason_codes:
                        import json as _json
                        try:
                            codes = _json.loads(reason_codes) if isinstance(reason_codes, str) else reason_codes
                            if codes:
                                st.markdown("**Karar Gerekçeleri:**")
                                for code in codes:
                                    st.markdown(f"  - `{code}`")
                        except Exception:
                            pass
                    # Retrieval detayları
                    retrieval_sources = selected_alert.get('retrieval_sources')
                    if retrieval_sources:
                        import json as _json
                        try:
                            src = _json.loads(retrieval_sources) if isinstance(retrieval_sources, str) else retrieval_sources
                            if src:
                                trgm_n = src.get('trgm', 0)
                                fts_n  = src.get('fts', 0)
                                vec_n  = src.get('vector', 0)
                                st.write(f"**Retrieval Kanalları:** Trigram={trgm_n} | FTS={fts_n} | Vector={vec_n} | Toplam Aday={selected_alert.get('candidate_count', '-')}")
                        except Exception:
                            pass
                    # Human explanation
                    human_exp = selected_alert.get('human_explanation')
                    if human_exp:
                        st.info(f"💬 **Sistem Açıklaması:** {human_exp}")
                    
                    with st.form(f"alert_review_form_{selected_alert_id}"):
                        analyst_name = st.text_input("Analist Adı", value=selected_alert.get('reviewed_by') or "")
                        new_status = st.selectbox("Durum", options=["OPEN", "IN_REVIEW", "CONFIRMED_MATCH", "FALSE_POSITIVE", "ESCALATED", "CLOSED"], index=["OPEN", "IN_REVIEW", "CONFIRMED_MATCH", "FALSE_POSITIVE", "ESCALATED", "CLOSED"].index(selected_alert['alert_status']) if selected_alert['alert_status'] in ["OPEN", "IN_REVIEW", "CONFIRMED_MATCH", "FALSE_POSITIVE", "ESCALATED", "CLOSED"] else 0)
                        review_result = st.text_input("İnceleme Sonucu (Özet)", value=selected_alert.get('review_result') or "")
                        analyst_note = st.text_area("Analist Notu", value=selected_alert.get('analyst_note') or "")
                        false_positive_reason = st.text_input("Yanlış Pozitif Nedeni (Eğer FP ise)", value=selected_alert.get('false_positive_reason') or "")
                        
                        submit_btn = st.form_submit_button("Durumu Güncelle")
                        if submit_btn:
                            repo = get_repo()
                            repo.update_alert_status(
                                alert_id=selected_alert_id, 
                                status=new_status, 
                                reviewed_by=analyst_name, 
                                review_result=review_result, 
                                analyst_note=analyst_note, 
                                false_positive_reason=false_positive_reason
                            )
                            st.success(f"Alert {selected_alert_id} başarıyla {new_status} olarak güncellendi!")
                            st.rerun()

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


# ── Benchmark sayfası ─────────────────────────────────────────────────────────
elif page == "🧪 Benchmark":
    st.title("🧪 Benchmark & Threshold Analizi")
    st.markdown("Pipeline performansını ölçmek ve threshold optimizasyonu yapmak için bu ekranı kullanın.")
    st.divider()

    tab_bm, tab_thresh = st.tabs(["📊 Benchmark Sonuçları", "📐 Threshold Analizi"])

    with tab_bm:
        st.subheader("📊 Benchmark Özeti")
        try:
            _repo = get_repo()
            _conn = _repo.get_connection()
            bm_df = pd.read_sql(f"""
                SELECT benchmark_run_name,
                       COUNT(*) AS n,
                       ROUND(AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END)::numeric, 3) AS accuracy,
                       ROUND(AVG(CASE WHEN recall_at_1 THEN 1.0 ELSE 0.0 END)::numeric, 3) AS recall_at_1,
                       ROUND(AVG(CASE WHEN recall_at_5 THEN 1.0 ELSE 0.0 END)::numeric, 3) AS recall_at_5,
                       ROUND(AVG(CASE WHEN recall_at_10 THEN 1.0 ELSE 0.0 END)::numeric, 3) AS recall_at_10,
                       ROUND(AVG(reciprocal_rank)::numeric, 3) AS mrr,
                       ROUND(AVG(processing_time_ms)::numeric, 1) AS avg_ms,
                       MAX(created_at) AS last_run
                FROM {TABLES['benchmark_result']}
                GROUP BY benchmark_run_name
                ORDER BY last_run DESC LIMIT 20
            """, _conn)
        except Exception:
            bm_df = pd.DataFrame()
        finally:
            try: _repo.release_connection(_conn)
            except: pass

        if bm_df.empty:
            st.info("Henüz benchmark sonucu yok. `src/evaluation/benchmark.py` ile çalıştırın.")
        else:
            st.dataframe(bm_df, use_container_width=True)
            recall_cols = ['recall_at_1', 'recall_at_5', 'recall_at_10']
            if all(c in bm_df.columns for c in recall_cols):
                fig_recall = px.bar(
                    bm_df, x='benchmark_run_name', y=recall_cols,
                    barmode='group',
                    title="Recall@K Karşılaştırması",
                    labels={'value': 'Recall', 'variable': 'K'},
                )
                fig_recall.update_layout(height=400)
                st.plotly_chart(fig_recall, use_container_width=True)

    with tab_thresh:
        st.subheader("📐 Threshold Analizi")
        st.info(
            "Threshold validasyonu için önce validator scriptini çalıştırın:\n\n"
            "`python -m src.evaluation.threshold_validator --scores outputs/scores.csv --output outputs/threshold_report.json`"
        )
        uploaded_report = st.file_uploader("Threshold Raporu Yükle (JSON)", type="json")
        if uploaded_report:
            import json as _json
            report = _json.load(uploaded_report)
            st.markdown(
                f"**Dataset:** {report.get('validation_dataset', 'N/A')} | "
                f"**N:** {report.get('n_samples', 0):,}"
            )
            cm = report.get('current_metrics', {})
            if cm:
                curr = report.get('current_thresholds', {})
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(f"Mevcut HIGH ({curr.get('high_threshold', '?')})", f"F1={cm.get('f1', 0):.3f}")
                c2.metric("Precision", f"{cm.get('precision', 0):.3f}")
                c3.metric("Recall", f"{cm.get('recall', 0):.3f}")
                c4.metric("FPR", f"{cm.get('fpr', 0):.3f}")
            best_f1 = report.get('recommendations', {}).get('best_f1', {})
            if best_f1:
                st.success(
                    f"**Önerilen (F1 Bazlı):** HIGH={best_f1.get('high_threshold')}, "
                    f"MEDIUM={best_f1.get('medium_threshold')} → F1={best_f1.get('f1', 0):.3f}"
                )
            st.warning(report.get('note', ''))
