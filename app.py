"""
Telecom Customer & Network Intelligence Platform
Main Streamlit Dashboard - app.py
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from generate_data import generate_telecom_dataset, generate_network_data, preprocess_data
from ml_models import (
    train_churn_model, train_segmentation_model,
    recommend_bundle, analyze_network,
    predict_churn_single, BUNDLE_CATALOG
)

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Intelligence Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    
    .main { background: #0a0e1a; }
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1f35, #0d1226);
        border: 1px solid rgba(100,160,255,0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #64a0ff; }
    .metric-label { font-size: 0.85rem; color: #8899bb; margin-top: 4px; }
    
    .section-header {
        background: linear-gradient(90deg, #1a2540, transparent);
        border-left: 4px solid #64a0ff;
        padding: 10px 20px;
        border-radius: 4px;
        margin: 20px 0 10px 0;
    }
    
    .risk-high { color: #ff4d6d; font-weight: 700; }
    .risk-medium { color: #ffb347; font-weight: 700; }
    .risk-low { color: #00d9a3; font-weight: 700; }
    
    .stButton>button {
        background: linear-gradient(135deg, #3a7bff, #1a4bcc);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 24px;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #4a8bff, #2a5bdd);
        transform: translateY(-1px);
    }
    
    .bundle-card {
        background: linear-gradient(135deg, #1a2540, #0d1830);
        border: 1px solid rgba(100,160,255,0.3);
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1226, #0a0e1a);
        border-right: 1px solid rgba(100,160,255,0.15);
    }
</style>
""", unsafe_allow_html=True)

# ─── Data & Model Loading (Cached) ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_all_data():
    df = generate_telecom_dataset(n_customers=2000)
    net_df = generate_network_data(n_records=5000)
    return df, net_df

@st.cache_resource(show_spinner=False)
def load_models():
    df, net_df = load_all_data()
    X, y, feature_cols, scaler, df_proc = preprocess_data(df)
    churn_results, best_churn_model, importances, splits = train_churn_model(X, y, feature_cols)
    kmeans, df_seg, cluster_profiles, sil_scores, sil, seg_names = train_segmentation_model(X, df_proc)
    net_analysis = analyze_network(net_df)
    return (
        df, net_df, X, y, feature_cols, scaler, df_proc,
        churn_results, best_churn_model, importances, splits,
        kmeans, df_seg, cluster_profiles, sil_scores, sil, seg_names,
        net_analysis
    )

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Telecom Intelligence")
    st.markdown("*ML-Powered Analytics Platform*")
    st.markdown("---")
    page = st.selectbox(
        "Navigate",
        ["🏠 Overview", "🔮 Churn Prediction", "👥 Customer Segments",
         "📦 Bundle Recommender", "🌐 Network Analytics", "🔍 Customer Lookup"]
    )
    st.markdown("---")
    st.markdown("**Dataset Info**")
    st.caption("2,000 synthetic customers\n5,000 network records\nBased on IBM Telco dataset")

# ─── LOAD DATA ─────────────────────────────────────────────────────────────────
with st.spinner("🔄 Initializing platform & training models..."):
    (df, net_df, X, y, feature_cols, scaler, df_proc,
     churn_results, best_churn_model, importances, splits,
     kmeans, df_seg, cluster_profiles, sil_scores, sil, seg_names,
     net_analysis) = load_models()

COLORS = ['#64a0ff', '#00d9a3', '#ffb347', '#ff4d6d', '#c084fc']

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📡 Telecom Customer & Network Intelligence Platform")
    st.caption("Data-driven insights powered by Machine Learning")
    st.markdown("---")

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    churn_rate = df['churn'].mean()
    avg_charges = df['monthly_charges'].mean()
    avg_tenure = df['tenure_months'].mean()
    avg_data = df['data_usage_gb'].mean()
    avg_congestion = net_analysis['avg_congestion']

    metrics = [
        (col1, f"{len(df):,}", "Total Customers"),
        (col2, f"{churn_rate:.1%}", "Churn Rate"),
        (col3, f"${avg_charges:.0f}", "Avg Monthly Charge"),
        (col4, f"{avg_tenure:.0f} mo", "Avg Tenure"),
        (col5, f"{avg_congestion:.0f}%", "Network Congestion"),
    ]
    for col, val, label in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📊 Churn by Contract Type")
        churn_by_contract = df.groupby('contract_type')['churn'].mean().reset_index()
        churn_by_contract['churn_pct'] = churn_by_contract['churn'] * 100
        fig = px.bar(churn_by_contract, x='contract_type', y='churn_pct',
                     color='churn_pct', color_continuous_scale='RdYlGn_r',
                     labels={'churn_pct': 'Churn Rate (%)', 'contract_type': 'Contract'},
                     template='plotly_dark')
        fig.update_layout(showlegend=False, height=300, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("📈 Customer Activity Distribution")
        fig = px.histogram(df, x='customer_activity_score', nbins=30,
                           color_discrete_sequence=['#64a0ff'],
                           template='plotly_dark',
                           labels={'customer_activity_score': 'Activity Score'})
        fig.update_layout(height=300, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("💳 Monthly Charges vs Data Usage")
        sample = df.sample(300, random_state=42)
        fig = px.scatter(sample, x='data_usage_gb', y='monthly_charges',
                         color='churn', color_discrete_map={0: '#00d9a3', 1: '#ff4d6d'},
                         template='plotly_dark', opacity=0.7,
                         labels={'data_usage_gb': 'Data Usage (GB)',
                                 'monthly_charges': 'Monthly Charges ($)',
                                 'churn': 'Churned'})
        fig.update_layout(height=300, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.subheader("🌍 Customers by Region")
        region_counts = df['region'].value_counts().reset_index()
        fig = px.pie(region_counts, values='count', names='region',
                     color_discrete_sequence=COLORS, template='plotly_dark',
                     hole=0.4)
        fig.update_layout(height=300, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CHURN PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Churn Prediction":
    st.title("🔮 Customer Churn Prediction")
    st.caption("Machine learning models to identify at-risk customers")
    st.markdown("---")

    # Model Performance
    st.subheader("📋 Model Performance Comparison")
    perf_data = []
    for name, r in churn_results.items():
        perf_data.append({
            'Model': name,
            'Accuracy': f"{r['accuracy']:.3f}",
            'Precision': f"{r['precision']:.3f}",
            'Recall': f"{r['recall']:.3f}",
            'F1-Score': f"{r['f1']:.3f}",
            'AUC-ROC': f"{r['auc_roc']:.3f}"
        })
    st.dataframe(pd.DataFrame(perf_data).set_index('Model'), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔑 Top Feature Importances")
        top_features = importances.head(12)
        fig = px.bar(top_features, x='importance', y='feature', orientation='h',
                     color='importance', color_continuous_scale='Blues',
                     template='plotly_dark')
        fig.update_layout(height=380, margin=dict(t=10), showlegend=False,
                          yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Churn Distribution")
        # Churn probability distribution using best model
        X_all_prob = best_churn_model.predict_proba(X)[:, 1]
        fig = px.histogram(x=X_all_prob, nbins=40,
                           color_discrete_sequence=['#64a0ff'],
                           template='plotly_dark',
                           labels={'x': 'Churn Probability'})
        fig.add_vline(x=0.35, line_dash="dash", line_color="#ffb347",
                      annotation_text="Medium Risk")
        fig.add_vline(x=0.65, line_dash="dash", line_color="#ff4d6d",
                      annotation_text="High Risk")
        fig.update_layout(height=380, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    # High-risk customers table
    st.subheader("⚠️ Top At-Risk Customers")
    df_risk = df_proc.copy()
    df_risk['churn_probability'] = best_churn_model.predict_proba(X)[:, 1]
    df_risk['risk_level'] = df_risk['churn_probability'].apply(
        lambda p: 'High' if p > 0.65 else ('Medium' if p > 0.35 else 'Low')
    )
    high_risk = df_risk[df_risk['risk_level'] == 'High'].nlargest(10, 'churn_probability')[
        ['customer_id', 'tenure_months', 'monthly_charges', 'complaint_count',
         'customer_activity_score', 'contract_type', 'churn_probability', 'risk_level']
    ].copy()
    high_risk['churn_probability'] = high_risk['churn_probability'].apply(lambda x: f"{x:.1%}")
    st.dataframe(high_risk, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CUSTOMER SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Segments":
    st.title("👥 Customer Segmentation")
    st.caption("K-Means clustering to identify customer groups")
    st.markdown("---")

    # Silhouette scores
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Segment Overview")
        seg_summary = df_seg.groupby('segment_name').agg(
            Count=('customer_id', 'count'),
            Avg_Monthly_Charge=('monthly_charges', 'mean'),
            Avg_Data_GB=('data_usage_gb', 'mean'),
            Avg_Tenure=('tenure_months', 'mean'),
            Churn_Rate=('churn', 'mean'),
            Avg_Activity=('customer_activity_score', 'mean')
        ).round(2)
        seg_summary['Churn_Rate'] = seg_summary['Churn_Rate'].apply(lambda x: f"{x:.1%}")
        seg_summary['Count'] = seg_summary['Count'].astype(int)
        st.dataframe(seg_summary, use_container_width=True)

    with col2:
        st.subheader("🎯 Silhouette Scores (k)")
        sil_df = pd.DataFrame({'k': list(sil_scores.keys()), 'Score': list(sil_scores.values())})
        fig = px.line(sil_df, x='k', y='Score', markers=True,
                      color_discrete_sequence=['#64a0ff'], template='plotly_dark')
        fig.update_layout(height=250, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Current Silhouette Score", f"{sil:.3f}")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🗂️ Segment Distribution")
        seg_counts = df_seg['segment_name'].value_counts().reset_index()
        fig = px.pie(seg_counts, values='count', names='segment_name',
                     color_discrete_sequence=COLORS, template='plotly_dark', hole=0.35)
        fig.update_layout(height=320, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("💡 Activity vs Monthly Charges")
        sample = df_seg.sample(400, random_state=1)
        fig = px.scatter(sample, x='customer_activity_score', y='monthly_charges',
                         color='segment_name', color_discrete_sequence=COLORS,
                         template='plotly_dark', opacity=0.75,
                         labels={'customer_activity_score': 'Activity Score',
                                 'monthly_charges': 'Monthly Charges ($)'})
        fig.update_layout(height=320, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    # Radar chart for segment profiles
    st.subheader("🕸️ Segment Profile Radar")
    metrics_radar = ['data_usage_gb', 'voice_minutes', 'monthly_charges',
                     'customer_activity_score', 'tenure_months']
    cluster_norm = cluster_profiles[metrics_radar].copy()
    for col in cluster_norm.columns:
        cluster_norm[col] = (cluster_norm[col] - cluster_norm[col].min()) / \
                             (cluster_norm[col].max() - cluster_norm[col].min() + 1e-9)

    fig = go.Figure()
    for i, (cluster_id, row) in enumerate(cluster_norm.iterrows()):
        seg_name = seg_names.get(cluster_id, f"Cluster {cluster_id}")
        vals = row.tolist() + [row.tolist()[0]]
        cats = metrics_radar + [metrics_radar[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill='toself',
            name=seg_name, line_color=COLORS[i % len(COLORS)],
            opacity=0.6
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template='plotly_dark', height=380, margin=dict(t=30)
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BUNDLE RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Bundle Recommender":
    st.title("📦 Bundle Recommendation System")
    st.caption("Personalized telecom bundle suggestions based on usage behavior")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Customer Profile")
        data_usage = st.slider("Data Usage (GB/month)", 0.0, 50.0, 8.0, 0.5)
        voice_mins = st.slider("Voice Minutes/month", 0, 800, 200, 10)
        sms = st.slider("SMS/month", 0, 500, 50, 10)
        monthly_charge = st.slider("Current Monthly Charge ($)", 5, 150, 45, 5)
        activity = st.slider("Activity Score", 0, 100, 55, 1)
        has_mm = st.checkbox("Has Mobile Money", value=True)
        has_stream = st.checkbox("Uses Streaming", value=False)

    with col2:
        st.subheader("🎯 Recommended Bundles")
        customer_profile = {
            'data_usage_gb': data_usage,
            'voice_minutes': voice_mins,
            'sms_count': sms,
            'monthly_charges': monthly_charge,
            'customer_activity_score': activity,
            'has_mobile_money': int(has_mm),
            'has_streaming': int(has_stream)
        }
        recs = recommend_bundle(customer_profile)

        for i, bundle in enumerate(recs):
            medal = ["🥇", "🥈", "🥉"][i]
            st.markdown(f"""
            <div class="bundle-card">
                <h4 style="color:#64a0ff; margin:0">{medal} {bundle['name']}</h4>
                <p style="color:#aabbcc; font-size:0.85rem; margin:4px 0">{bundle['description']}</p>
                <div style="display:flex; gap:20px; margin-top:8px">
                    <span>💰 <b>${bundle['price']}/mo</b></span>
                    <span>📶 <b>{bundle['data_gb']} GB</b></span>
                    <span>📞 <b>{bundle['minutes']} min</b></span>
                    <span>💬 <b>{bundle['sms']} SMS</b></span>
                </div>
                <div style="margin-top:6px">
                    <span style="color:#00d9a3; font-size:0.8rem">Match Score: {bundle['score']:.0f}/100</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Bundle catalog overview
    st.markdown("---")
    st.subheader("📋 Full Bundle Catalog")
    catalog_df = pd.DataFrame(BUNDLE_CATALOG).T.reset_index()
    catalog_df.columns = ['Bundle', 'Price ($)', 'Data (GB)', 'Minutes', 'SMS', 'Description']
    catalog_df['Price ($)'] = catalog_df['Price ($)'].astype(float)
    catalog_df['Data (GB)'] = catalog_df['Data (GB)'].astype(float)
    catalog_df['Minutes'] = catalog_df['Minutes'].astype(float)
    fig = px.bar(catalog_df, x='Bundle', y='Price ($)',
                 color='Data (GB)', color_continuous_scale='Blues',
                 template='plotly_dark',
                 text='Price ($)',
                 labels={'Price ($)': 'Monthly Price ($)', 'Bundle': 'Bundle Name'})
    fig.update_traces(textposition='outside')
    fig.update_layout(height=380, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: NETWORK ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌐 Network Analytics":
    st.title("🌐 Network Congestion Analytics")
    st.caption("Traffic pattern analysis and congestion detection")
    st.markdown("---")

    # KPIs
    kpi_cols = st.columns(4)
    kpis = [
        ("Avg Congestion", f"{net_analysis['avg_congestion']}%"),
        ("Avg Latency", f"{net_analysis['avg_latency']} ms"),
        ("Avg Packet Loss", f"{net_analysis['avg_packet_loss']}%"),
        ("Peak Hours", ", ".join([f"{h}:00" for h in net_analysis['peak_hours']])),
    ]
    for col, (label, val) in zip(kpi_cols, kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size:1.4rem">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🕐 Hourly Congestion Pattern")
        hourly = net_analysis['hourly'].reset_index()
        fig = px.area(hourly, x='hour', y='congestion_level',
                      color_discrete_sequence=['#64a0ff'], template='plotly_dark',
                      labels={'hour': 'Hour of Day', 'congestion_level': 'Congestion (%)'},
                      line_shape='spline')
        fig.update_layout(height=300, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📅 Daily Congestion Heatmap")
        daily = net_analysis['daily'].reset_index()
        fig = px.bar(daily, x='day', y='congestion_level',
                     color='congestion_level', color_continuous_scale='RdYlGn_r',
                     template='plotly_dark',
                     labels={'congestion_level': 'Avg Congestion (%)', 'day': 'Day'})
        fig.update_layout(height=300, margin=dict(t=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🗺️ Regional Network Performance")
        regional = net_analysis['regional'].reset_index()
        fig = px.bar(regional, x='region', y=['congestion_level', 'latency_ms'],
                     barmode='group', template='plotly_dark',
                     color_discrete_sequence=['#64a0ff', '#ff4d6d'],
                     labels={'value': 'Level', 'region': 'Region', 'variable': 'Metric'})
        fig.update_layout(height=320, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("📶 Traffic vs Latency Correlation")
        sample_net = net_df.sample(500, random_state=42)
        fig = px.scatter(sample_net, x='traffic_mbps', y='latency_ms',
                         color='congestion_level', color_continuous_scale='RdYlGn_r',
                         template='plotly_dark', opacity=0.6,
                         labels={'traffic_mbps': 'Traffic (Mbps)', 'latency_ms': 'Latency (ms)'})
        fig.update_layout(height=320, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    # Congestion alerts
    if net_analysis['high_congestion_regions']:
        st.warning(f"⚠️ High Congestion Regions: {', '.join(net_analysis['high_congestion_regions'])}")
    else:
        st.success("✅ No regions currently exceeding congestion threshold")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CUSTOMER LOOKUP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Customer Lookup":
    st.title("🔍 Individual Customer Analysis")
    st.caption("Look up any customer for churn risk, segment, and bundle recommendation")
    st.markdown("---")

    customer_ids = df_proc['customer_id'].tolist()
    selected_id = st.selectbox("Select Customer ID", customer_ids)

    if selected_id:
        cust = df_proc[df_proc['customer_id'] == selected_id].iloc[0]
        cust_seg = df_seg[df_seg['customer_id'] == selected_id].iloc[0]

        # Predict churn
        idx = df_proc[df_proc['customer_id'] == selected_id].index[0]
        churn_prob = best_churn_model.predict_proba(X[idx:idx+1])[0][1]
        risk = 'High' if churn_prob > 0.65 else ('Medium' if churn_prob > 0.35 else 'Low')
        risk_color = {'High': '#ff4d6d', 'Medium': '#ffb347', 'Low': '#00d9a3'}[risk]

        col1, col2, col3 = st.columns(3)
        col1.metric("Churn Probability", f"{churn_prob:.1%}")
        col2.metric("Risk Level", risk)
        col3.metric("Segment", cust_seg['segment_name'])

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("👤 Customer Profile")
            profile = {
                "Age": cust['age'],
                "Gender": cust['gender'],
                "Tenure": f"{cust['tenure_months']} months",
                "Contract": cust['contract_type'],
                "Region": cust['region'],
                "Monthly Charge": f"${cust['monthly_charges']:.2f}",
                "Data Usage": f"{cust['data_usage_gb']:.1f} GB",
                "Voice Minutes": cust['voice_minutes'],
                "Complaints": cust['complaint_count'],
                "Activity Score": f"{cust['customer_activity_score']:.1f}/100"
            }
            for k, v in profile.items():
                st.write(f"**{k}:** {v}")

        with col_b:
            st.subheader("📦 Recommended Bundles")
            recs = recommend_bundle(cust.to_dict())
            for i, bundle in enumerate(recs):
                medal = ["🥇", "🥈", "🥉"][i]
                st.markdown(f"""
                <div class="bundle-card">
                    <b>{medal} {bundle['name']}</b> — ${bundle['price']}/mo<br>
                    <span style="color:#aabb99; font-size:0.8rem">
                        {bundle['data_gb']} GB · {bundle['minutes']} min · {bundle['sms']} SMS
                    </span>
                </div>""", unsafe_allow_html=True)

        # Churn risk gauge
        st.subheader("🎯 Churn Risk Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Risk %", 'font': {'color': 'white'}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 35], 'color': '#0d2b1a'},
                    {'range': [35, 65], 'color': '#2b2000'},
                    {'range': [65, 100], 'color': '#2b0a0a'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 2},
                    'thickness': 0.75,
                    'value': churn_prob * 100
                }
            },
            number={'suffix': "%", 'font': {'color': risk_color, 'size': 40}}
        ))
        fig.update_layout(template='plotly_dark', height=280, margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
)
