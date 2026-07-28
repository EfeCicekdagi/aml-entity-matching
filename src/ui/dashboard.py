"""
AML Alert Review Dashboard — Streamlit
Calistirilmak icin: streamlit run src/ui/dashboard.py
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import pandas as pd
import plotly.express as px
from src.config.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository
from src.utils.text_utils import normalize_text, get_normalized_core_name, is_consonant_match
from src.config.db_tables import TABLES
from src.scoring.score_features import _acronym_score, _rule_score, _exact_name_score, build_score_features

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
    return AMLRepository(
        host=cfg['host'], port=cfg['port'], dbname=cfg['name'],
        user=cfg['user'], password=cfg['password'],
        sslmode=cfg.get('sslmode', 'prefer'),
        enable_audit_trail=cfg.get('enable_audit_trail', True),
        append_only_history=cfg.get('append_only_history', True)
    )

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
        # candidate_rank = 1: her EFT için sadece en yüksek skorlu eşleşmeyi göster.
        # Eski veriler için (candidate_rank NULL) DISTINCT ON fallback uygulanır.
        df = pd.read_sql(f"""
            SELECT DISTINCT ON (eft_id) *
            FROM {TABLES['alert_export']}
            WHERE run_id = %(run_id)s
              AND (candidate_rank = 1 OR candidate_rank IS NULL)
            ORDER BY eft_id, final_score DESC
        """, conn, params={"run_id": run_id})
    finally:
        repo.release_connection(conn)
    return df

@st.cache_resource
def load_live_components():
    import torch
    from sentence_transformers import SentenceTransformer
    from src.models.ner_extractor import NERExtractor
    from src.retrieval.postgres_candidate_retriever import PostgresCandidateRetriever
    from src.models.reranker import Reranker
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
    
    page = st.radio("Menü", ["🏠 Ana Sayfa", "📈 Run Detayları", "📊 Model Optimizasyon"])
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
                        cand_copy = dict(cand)
                        if "pg_trgm" in cand.get("sources", []):
                            cand_copy["trgm_score"] = cand.get("candidate_score", 0.0)
                        if "pgvector" in cand.get("sources", []):
                            cand_copy["vector_score"] = cand.get("candidate_score", 0.0)
                        scores_dict = build_score_features(norm_exp, cand_copy, extracted_entity=entity, raw_explanation=user_input)
                        final_score, match_reason, reason_codes = scorer.calculate_final_score(scores_dict)
                        risk_level  = scorer.assign_risk_level(final_score)
                        
                        results.append({
                            "Şirket": cand["company_name"],
                            "Varyant": cand["variant_name"],
                            "Risk": risk_level,
                            "Sebep": match_reason,
                            "Gerekçeler": ", ".join(reason_codes),
                            "Final Skor": final_score,
                            "Reranker Skor": scores_dict["reranker_score"],
                            "Vektör Skor": scores_dict["vector_score"],
                            "Fuzzy Skor": scores_dict["fuzzy_score"],
                            "Exact Compact Match": "Evet" if scores_dict["exact_compact_match"] else "Hayır",
                            "Compact Variant": scores_dict["compact_matched_variant"]
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

    # ── KPI Metrikleri (EFT Bazlı) ────────────────────────────────────────────────
    run_row = runs_df[runs_df['run_id'] == selected_run].iloc[0]
    total_efts = int(run_row.get('input_row_count', 0))
    no_cand_efts = int(run_row.get('no_candidate_count', 0))
    
    # Alert alan EFT'lerin benzersiz sayılarını bul (Öncelik HIGH)
    high_eft_ids = set(alerts_df[alerts_df['risk_level']=='HIGH']['eft_id']) if not alerts_df.empty else set()
    med_eft_ids = set(alerts_df[alerts_df['risk_level']=='MEDIUM']['eft_id']) if not alerts_df.empty else set()
    
    # Eğer bir EFT hem HIGH hem MEDIUM alert aldıysa, onu sadece HIGH say!
    med_eft_ids = med_eft_ids - high_eft_ids
    
    high_efts = len(high_eft_ids)
    medium_efts = len(med_eft_ids)
    
    # LOW EFT = Toplam EFT - (HIGH + MEDIUM + No Candidate)
    low_efts = max(0, total_efts - (high_efts + medium_efts + no_cand_efts))
    
    # İlk Satır (4 Kutu)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Toplam İşlenen EFT", f"{total_efts:,}")
    with col2:
        st.metric("🔴 HIGH (EFT)", f"{high_efts:,}")
    with col3:
        st.metric("🟠 MEDIUM (EFT)", f"{medium_efts:,}")
    with col4:
        st.metric("🟢 LOW (EFT)", f"{low_efts:,}")
        
    st.markdown("<br>", unsafe_allow_html=True) # Araya boşluk
    
    # İkinci Satır (4 Kutu)
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        avg_score = alerts_df['final_score'].mean() if not alerts_df.empty else 0.0
        st.metric("Ortalama Alert Skoru", f"{avg_score:.3f}")
    with col6:
        st.metric("⚫ No Candidate (EFT)", f"{no_cand_efts:,}")
    with col7:
        p95 = run_row.get('p95_latency_ms', None) if not runs_df.empty else None
        st.metric("P95 Latency", f"{p95:.0f} ms" if p95 and p95 > 0 else "N/A")
    with col8:
        st.metric("Üretilen Toplam Alert", f"{len(filtered):,}")

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
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dağılım Analizi", "📋 Alert Listesi", "🏢 Şirket Analizi", "⚖️ Vektör vs Fuzzy (AI Sapma)"])

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
                fig.add_vline(x=0.60, line_dash="dash", line_color="#f97316",
                              annotation_text="MEDIUM eşiği (0.60)")
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

    with tab4:
        st.subheader("⚖️ Vektör (Yapay Zeka) vs. Fuzzy (Karakter) Karşılaştırmalı Sapma Analizi")
        st.markdown("Bu ekranda **BGE-M3 Vektör (AI)** modeli ile **Fuzzy (Karakter)** algoritması arasındaki skorsal sapmaları analiz edebilirsiniz. **Çizginin üstünde kalanlar yapay zekanın yakaladıkları (anlamsal eşleşmeler), çizginin altında kalanlar karakter algoritmamızın yakaladıklarıdır (tuzak ve kamuflaj koruması).**")
        
        if alerts_df.empty:
            st.info("Analiz yapılabilecek bir alert verisi bulunamadı.")
        else:
            # 1. Scatter Plot (Saçılım Grafiği)
            fig_div = px.scatter(
                alerts_df, x='fuzzy_score', y='vector_score', color='risk_level',
                color_discrete_map={"HIGH":"#ef4444","MEDIUM":"#f97316","LOW":"#6b7280"},
                hover_data=['original_company_name', 'matched_variant_name', 'original_explanation'],
                title="Vektör vs. Fuzzy Skoru Matrisi (Diyagonal Üstü: Yapay Zeka Gücü | Altı: Karakter & Kamuflaj)",
                labels={'fuzzy_score':'Fuzzy (Karakter Benzerliği) Skoru', 'vector_score':'Vektör (AI Anlamsal) Skoru'},
                opacity=0.85
            )
            fig_div.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="White", width=2, dash="dash"))
            fig_div.update_layout(height=450)
            st.plotly_chart(fig_div, use_container_width=True)
            
            st.divider()
            
            # 2. Etkileşimli Eşik Slider'ı
            min_sapma = st.slider("🔍 Minimum Sapma (Skor Farkı) Eşiği:", min_value=0.05, max_value=0.50, value=0.15, step=0.05, help="İki model arasındaki minimum puan farkını belirleyerek tabloları filtreleyin.")
            
            sub1, sub2 = st.tabs(["🟢 Vektör > Fuzzy (Yapay Zeka Yakalamaları)", "🔵 Fuzzy > Vektör (Karakter Gücü & Kamuflaj)"])
            
            with sub1:
                st.markdown("**💡 Yapay Zekanın Gücü:** Harfler benzemediği halde (düşük Fuzzy), anlamsal veya sektörel olarak eşleştiği için Vektör yapay zekamızın yüksek skor verdiği işlemler.")
                v_gt_f = alerts_df[alerts_df['vector_score'] - alerts_df['fuzzy_score'] >= min_sapma].copy()
                if not v_gt_f.empty:
                    v_gt_f['sapma'] = (v_gt_f['vector_score'] - v_gt_f['fuzzy_score']).round(3)
                    v_gt_f = v_gt_f.sort_values('sapma', ascending=False)
                    st.success(f"**{len(v_gt_f)}** adet işlemde Vektör skoru Fuzzy skorundan `{min_sapma}` puan veya daha fazla yüksektir.")
                    st.dataframe(v_gt_f[['eft_id', 'original_explanation', 'matched_variant_name', 'vector_score', 'fuzzy_score', 'reranker_score', 'final_score', 'risk_level', 'sapma']].rename(columns={
                        'eft_id':'EFT ID', 'original_explanation':'EFT Açıklaması', 'matched_variant_name':'Eşleşen Şirket',
                        'vector_score':'Vektör', 'fuzzy_score':'Fuzzy', 'reranker_score':'Reranker', 'final_score':'Final Skor',
                        'risk_level':'Risk', 'sapma':'Fark (+)'
                    }), use_container_width=True)
                else:
                    st.info(f"Bu eşikte (`>={min_sapma}`) Vektör > Fuzzy sapma kaydı bulunamadı. Eşiği düşürebilirsiniz.")

            with sub2:
                st.markdown("**💡 Karakter & Normalizasyon Gücü:** Harf benzerliği yüksek olup (yüksek Fuzzy), leetspeak/kamuflaj yakalamaları veya Vektörün farklı sektör diye ayırt ettiği işlemler.")
                f_gt_v = alerts_df[alerts_df['fuzzy_score'] - alerts_df['vector_score'] >= min_sapma].copy()
                if not f_gt_v.empty:
                    f_gt_v['sapma'] = (f_gt_v['fuzzy_score'] - f_gt_v['vector_score']).round(3)
                    f_gt_v = f_gt_v.sort_values('sapma', ascending=False)
                    st.success(f"**{len(f_gt_v)}** adet işlemde Fuzzy skoru Vektör skorundan `{min_sapma}` puan veya daha fazla yüksektir.")
                    st.dataframe(f_gt_v[['eft_id', 'original_explanation', 'matched_variant_name', 'fuzzy_score', 'vector_score', 'reranker_score', 'final_score', 'risk_level', 'sapma']].rename(columns={
                        'eft_id':'EFT ID', 'original_explanation':'EFT Açıklaması', 'matched_variant_name':'Eşleşen Şirket',
                        'fuzzy_score':'Fuzzy', 'vector_score':'Vektör', 'reranker_score':'Reranker', 'final_score':'Final Skor',
                        'risk_level':'Risk', 'sapma':'Fark (+)'
                    }), use_container_width=True)
                else:
                    st.info(f"Bu eşikte (`>={min_sapma}`) Fuzzy > Vektör sapma kaydı bulunamadı. Eşiği düşürebilirsiniz.")


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

elif page == "📊 Model Optimizasyon":
    st.title("📊 Model Optimizasyon Raporu")
    st.markdown("Bu sayfada son yapılan ağırlık (weight) ve eşik (threshold) optimizasyonu testlerinin sonuçlarını ve seçilen konfigürasyonların matematiksel gerekçelerini (kanıtlarını) inceleyebilirsiniz.")
    
    st.divider()
    
    repo = get_repo()
    conn = repo.get_connection()
    if not conn:
        st.error("Veritabanına bağlanılamadı.")
    else:
        try:
            # 1. Weight Optimizasyonu Sonuçları
            st.markdown("### 1. Ağırlık (Weight) Optimizasyonu Sonuçları")
            st.markdown("Farklı (Fuzzy, Vector, Reranker) ağırlık kombinasyonlarının **F1 Skoru** üzerindeki etkileri:")
            
            weight_df = pd.read_sql("""
                SELECT fuzzy_weight, vector_weight, reranker_weight, f1_score, precision_score, recall_score
                FROM aml_experiment.weight_analysis
                WHERE experiment_id = (
                    SELECT experiment_id FROM aml_experiment.weight_analysis 
                    GROUP BY experiment_id 
                    HAVING count(*) >= 10 
                    ORDER BY experiment_id DESC 
                    LIMIT 1
                )
                ORDER BY f1_score DESC
            """, conn)
            
            if not weight_df.empty:
                weight_df = weight_df.drop_duplicates(subset=["fuzzy_weight", "vector_weight", "reranker_weight"]).reset_index(drop=True)
                weight_df["Model Konfigürasyonu (F/V/R)"] = weight_df.apply(
                    lambda row: f"F:{row['fuzzy_weight']:.1f} V:{row['vector_weight']:.1f} R:{row['reranker_weight']:.1f}", axis=1)
                
                best_w = weight_df.iloc[0]
                st.success(f"**🌟 En İyi Konfigürasyon:** **Fuzzy: `{best_w['fuzzy_weight']:.2f}` | Vector: `{best_w['vector_weight']:.2f}` | Reranker: `{best_w['reranker_weight']:.2f}`**\n\n"
                           f"• **F1 Skoru:** `{best_w['f1_score']:.4f}` | **Recall:** `{best_w['recall_score']:.4f}` | **Precision:** `{best_w['precision_score']:.4f}`\n\n"
                           f"*Bu konfigürasyon F1 skorunu maksimize ettiği için sistemin aktif varsayılan ağırlıkları olarak seçilmiştir.*")
                
                # Tüm denemeleri (kombinasyonları) bar chart ile göster
                all_weights_sorted = weight_df.sort_values("f1_score", ascending=True)
                fig1 = px.bar(all_weights_sorted, x="f1_score", y="Model Konfigürasyonu (F/V/R)", orientation='h',
                              title="Tüm Ağırlık Konfigürasyonlarının F1 Skoru Karşılaştırması (Tam Liste)",
                              color="f1_score", color_continuous_scale="Viridis",
                              labels={'f1_score': 'F1 Skoru'})
                fig1.update_layout(height=800)
                st.plotly_chart(fig1, use_container_width=True)
                
                # Tüm kombinasyon listesini doğrudan, gizlemeden açıkça gösterelim
                st.markdown("#### 📋 Tüm Ağırlık Kombinasyonları Karşılaştırma Listesi (Tam Tablo)")
                st.markdown("*Aşağıdaki tabloda denenen **tüm ağırlık kombinasyonları** (örneğin 0.1 / 0.1 / 0.8 gibi) F1 Skoruna göre sıralanmıştır. Sütun başlıklarına tıklayarak sıralamayı değiştirebilir veya arama yapabilirsiniz.*")
                
                display_df = weight_df[["fuzzy_weight", "vector_weight", "reranker_weight", "f1_score", "precision_score", "recall_score"]].copy()
                display_df = display_df.rename(columns={
                    "fuzzy_weight": "Fuzzy Ağırlığı",
                    "vector_weight": "Vector Ağırlığı",
                    "reranker_weight": "Reranker Ağırlığı",
                    "f1_score": "F1 Skoru",
                    "precision_score": "Kesinlik (Precision)",
                    "recall_score": "Duyarlılık (Recall)"
                })
                
                st.dataframe(
                    display_df.style.format({
                        "Fuzzy Ağırlığı": "{:.2f}",
                        "Vector Ağırlığı": "{:.2f}",
                        "Reranker Ağırlığı": "{:.2f}",
                        "F1 Skoru": "{:.4f}",
                        "Kesinlik (Precision)": "{:.4f}",
                        "Duyarlılık (Recall)": "{:.4f}"
                    }).highlight_max(subset=["F1 Skoru", "Kesinlik (Precision)", "Duyarlılık (Recall)"], color="#d4edda")
                      .highlight_min(subset=["F1 Skoru"], color="#f8d7da"),
                    use_container_width=True,
                    height=450
                )
            else:
                st.info("Ağırlık optimizasyonu verisi bulunamadı.")
                
            st.divider()
            
            # 2. Threshold Optimizasyonu Sonuçları
            st.markdown("### 2. Eşik (Threshold) Optimizasyonu Sonuçları")
            st.markdown("Reranker ağırlığı sabitlendiğinde (en iyi konfigürasyonda), farklı risk eşiklerinin yakalama oranları (Recall) ve kesinlik (Precision) metrikleri:")
            
            thresh_df = pd.read_sql("""
                SELECT medium_threshold, high_threshold, f1_score, precision_score, recall_score
                FROM aml_experiment.threshold_analysis
                WHERE experiment_id = (
                    SELECT experiment_id FROM aml_experiment.threshold_analysis 
                    GROUP BY experiment_id 
                    HAVING count(*) >= 5 
                    ORDER BY experiment_id DESC 
                    LIMIT 1
                )
                ORDER BY f1_score DESC
            """, conn)
            
            if not thresh_df.empty:
                thresh_df = thresh_df.drop_duplicates(subset=["high_threshold", "medium_threshold"]).reset_index(drop=True)
                thresh_df["Eşik Seçimi (High/Medium)"] = thresh_df.apply(
                    lambda row: f"H:{row['high_threshold']:.2f} / M:{row['medium_threshold']:.2f}", axis=1)
                
                all_thresh_sorted = thresh_df.sort_values("f1_score", ascending=True)
                fig2 = px.scatter(thresh_df, x="recall_score", y="precision_score", 
                                  color="f1_score", hover_name="Eşik Seçimi (High/Medium)",
                                  title="Precision vs Recall - Eşik Analizi",
                                  labels={'recall_score': 'Duyarlılık (Recall)', 'precision_score': 'Kesinlik (Precision)'})
                st.plotly_chart(fig2, use_container_width=True)
                
                fig3 = px.bar(all_thresh_sorted, x="f1_score", y="Eşik Seçimi (High/Medium)", orientation='h',
                              title="Tüm Eşik Konfigürasyonlarının F1 Skoru Karşılaştırması (Tam Liste)",
                              color="f1_score", color_continuous_scale="Blues")
                fig3.update_layout(height=700)
                st.plotly_chart(fig3, use_container_width=True)
                
                st.markdown("#### 📋 Tüm Eşik Testleri Karşılaştırma Listesi (Tam Tablo)")
                st.markdown("*Aşağıdaki tabloda denenen **tüm risk eşiği kombinasyonları** (High / Medium) F1 Skoruna göre sıralanmıştır.*")
                
                t_display_df = thresh_df[["high_threshold", "medium_threshold", "f1_score", "precision_score", "recall_score"]].copy()
                t_display_df = t_display_df.rename(columns={
                    "high_threshold": "High Eşik (Yüksek Risk)",
                    "medium_threshold": "Medium Eşik (Orta Risk)",
                    "f1_score": "F1 Skoru",
                    "precision_score": "Kesinlik (Precision)",
                    "recall_score": "Duyarlılık (Recall)"
                })
                
                st.dataframe(
                    t_display_df.style.format({
                        "High Eşik (Yüksek Risk)": "{:.2f}",
                        "Medium Eşik (Orta Risk)": "{:.2f}",
                        "F1 Skoru": "{:.4f}",
                        "Kesinlik (Precision)": "{:.4f}",
                        "Duyarlılık (Recall)": "{:.4f}"
                    }).highlight_max(subset=["F1 Skoru", "Kesinlik (Precision)", "Duyarlılık (Recall)"], color="#d4edda")
                      .highlight_min(subset=["F1 Skoru"], color="#f8d7da"),
                    use_container_width=True,
                    height=450
                )
                    
                best_f1 = thresh_df.iloc[0]
                best_recall_df = thresh_df[thresh_df["precision_score"] >= 0.85].sort_values(by=["recall_score", "f1_score"], ascending=False)
                best_rec = best_recall_df.iloc[0] if not best_recall_df.empty else best_f1
                best_prec_df = thresh_df[thresh_df["recall_score"] >= 0.40].sort_values(by=["precision_score", "f1_score"], ascending=False)
                best_prec = best_prec_df.iloc[0] if not best_prec_df.empty else best_f1

                st.markdown("#### 🎯 Eşik Stratejisi Karar Matrisi & Öneriler")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.success(f"**⚖️ En Optimize (Best F1)**\n\n"
                               f"**High:** `{best_f1['high_threshold']:.2f}` | **Medium:** `{best_f1['medium_threshold']:.2f}`\n\n"
                               f"• **F1 Skoru:** `{best_f1['f1_score']:.4f}`\n"
                               f"• **Recall:** `{best_f1['recall_score']:.4f}`\n"
                               f"• **Precision:** `{best_f1['precision_score']:.4f}`")
                with col2:
                    st.info(f"**🛡️ En Yüksek Recall (AML Önerisi)**\n*(Precision ≥ %85 şartıyla)*\n\n"
                            f"**High:** `{best_rec['high_threshold']:.2f}` | **Medium:** `{best_rec['medium_threshold']:.2f}`\n\n"
                            f"• **F1 Skoru:** `{best_rec['f1_score']:.4f}`\n"
                            f"• **Recall:** `{best_rec['recall_score']:.4f}`\n"
                            f"• **Precision:** `{best_rec['precision_score']:.4f}`")
                with col3:
                    st.warning(f"**🎯 En Yüksek Precision (Az Alarm)**\n*(Recall ≥ %40 şartıyla)*\n\n"
                               f"**High:** `{best_prec['high_threshold']:.2f}` | **Medium:** `{best_prec['medium_threshold']:.2f}`\n\n"
                               f"• **F1 Skoru:** `{best_prec['f1_score']:.4f}`\n"
                               f"• **Recall:** `{best_prec['recall_score']:.4f}`\n"
                               f"• **Precision:** `{best_prec['precision_score']:.4f}`")
            else:
                st.info("Threshold optimizasyonu verisi bulunamadı.")
                
        except Exception as e:
            st.error(f"Veri çekilirken hata oluştu: {e}")
        finally:
            repo.release_connection(conn)
