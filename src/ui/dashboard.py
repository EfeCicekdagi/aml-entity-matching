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
    df = pd.read_sql("""
        SELECT run_id, started_at, finished_at,
               ROUND(EXTRACT(EPOCH FROM (finished_at-started_at))/60,1) AS sure_dk,
               input_row_count, candidate_count, alert_count, status
        FROM aml_run_log ORDER BY started_at DESC LIMIT 20
    """, conn)
    repo.release_connection(conn)
    return df

@st.cache_data(ttl=30)
def load_alerts(run_id):
    repo = get_repo()
    conn = repo.get_connection()
    df = pd.read_sql("""
        SELECT a.alert_id, a.eft_id, v.original_company_name,
               ROUND(a.final_score::numeric, 3) AS final_score,
               a.risk_level, a.alert_status, a.extracted_entity, a.created_at
        FROM aml_alert a
        JOIN silver_company_variant v ON a.variant_id=v.variant_id
        WHERE a.run_id = %(run_id)s
        ORDER BY a.final_score DESC
    """, conn, params={"run_id": run_id})
    repo.release_connection(conn)
    return df

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚨 AML Dashboard")
    st.divider()

    runs_df = load_runs()
    if runs_df.empty:
        st.error("Hicbir run bulunamadi.")
        st.stop()

    run_options = runs_df['run_id'].tolist()
    selected_run = st.selectbox("Run Seç", run_options,
                                 format_func=lambda x: f"{x}  ({runs_df[runs_df.run_id==x]['started_at'].values[0]})")

    st.divider()
    run_info = runs_df[runs_df['run_id'] == selected_run].iloc[0]
    st.markdown(f"**Süre:** {run_info['sure_dk']} dk")
    st.markdown(f"**Girdi:** {run_info['input_row_count']:,} EFT")
    st.markdown(f"**Aday:** {run_info['candidate_count']:,}")
    st.markdown(f"**Durum:** {'✅' if run_info['status']=='SUCCESS' else '❌'} {run_info['status']}")

    st.divider()
    risk_filter = st.multiselect("Risk Filtresi", ["HIGH", "MEDIUM", "LOW"],
                                  default=["HIGH", "MEDIUM"])
    score_min = st.slider("Min Final Score", 0.0, 1.0, 0.50, 0.01)
    st.divider()
    if st.button("🔄 Yenile"):
        st.cache_data.clear()
        st.rerun()

# ── Ana icerik ────────────────────────────────────────────────────────────────
st.title(f"🚨 AML Alert Review — {selected_run}")

alerts_df = load_alerts(selected_run)
if alerts_df.empty:
    st.warning("Bu run icin alert bulunamadi.")
    st.stop()

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
    high_cnt = len(alerts_df[alerts_df['risk_level']=='HIGH'])
    st.metric("🔴 HIGH", f"{high_cnt:,}")
with col3:
    med_cnt = len(alerts_df[alerts_df['risk_level']=='MEDIUM'])
    st.metric("🟠 MEDIUM", f"{med_cnt:,}")
with col4:
    st.metric("Ortalama Skor", f"{alerts_df['final_score'].mean():.3f}")
with col5:
    st.metric("Gösterilen", f"{len(filtered):,}")

st.divider()

# ── Grafikler ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Dağılım Analizi", "📋 Alert Listesi", "🏢 Şirket Analizi"])

with tab1:
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

    # Run karsilastirma
    all_runs = load_runs()
    fig3 = px.bar(all_runs.sort_values('started_at'), x='run_id', y='alert_count',
                  title="Run Bazında Alert Sayısı Trendi",
                  text='alert_count', color='alert_count',
                  color_continuous_scale='Reds')
    fig3.update_traces(textposition='outside')
    fig3.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.markdown(f"**{len(filtered):,} alert gösteriliyor** (filtre: risk={risk_filter}, score≥{score_min})")

    # Risk badge fonksiyonu
    def badge(risk):
        colors = {"HIGH":"🔴","MEDIUM":"🟠","LOW":"⚪"}
        return f"{colors.get(risk,'')} {risk}"

    display_df = filtered[['eft_id','original_company_name','extracted_entity','final_score','risk_level','alert_status','created_at']].copy()
    display_df['risk_level'] = display_df['risk_level'].apply(badge)
    display_df.columns = ['EFT ID','Şirket','Çıkarılan İsim (NER)','Final Score','Risk','Durum','Tarih']

    st.dataframe(display_df, use_container_width=True, height=500,
                 column_config={
                     "Final Score": st.column_config.ProgressColumn(
                         "Final Score", min_value=0, max_value=1, format="%.3f"
                     )
                 })

    # CSV indirme
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button("📥 CSV İndir", csv, f"alerts_{selected_run}.csv", "text/csv")

with tab3:
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
