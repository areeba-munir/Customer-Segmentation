import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="Executive Customer Analytics", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# Clean, Modern SaaS Light Theme CSS
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #f4f7f6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        margin-bottom: 30px;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2.5rem; font-weight: 700; }
    .main-header p { margin: 5px 0 0 0; font-size: 1.1rem; opacity: 0.9; }
    
    /* Custom Metric Cards */
    div[data-testid="metric-container"] {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #3B82F6;
        transition: transform 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Chart Containers */
    .chart-box {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    df.rename(columns={'Annual Income (k$)': 'Income', 'Spending Score (1-100)': 'Spending_Score'}, inplace=True)
    return df

df = load_data()

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3126/3126647.png", width=80)
    st.title("⚙️ Model Controls")
    
    st.markdown("### Clustering Parameters")
    n_clusters = st.slider("Target Segments (K)", min_value=3, max_value=8, value=5)
    
    st.markdown("### Data Filters")
    gender_filter = st.multiselect("Select Gender", options=["Male", "Female"], default=["Male", "Female"])
    
    # Filter Data
    filtered_df = df[df['Gender'].isin(gender_filter)].copy()

# ==========================================
# 4. MACHINE LEARNING: K-MEANS & PERSONA LOGIC
# ==========================================
if not filtered_df.empty:
    X = filtered_df[['Income', 'Spending_Score']]
    
    # Fit Model
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    filtered_df['Cluster'] = kmeans.fit_predict(X)
    
    # DYNAMIC PERSONA GENERATOR: Analyzes cluster centers to name them!
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['Income', 'Spending_Score'])
    inc_mean, spend_mean = X['Income'].mean(), X['Spending_Score'].mean()
    
    persona_map = {}
    for i, row in centroids.iterrows():
        if row['Income'] > inc_mean and row['Spending_Score'] > spend_mean:
            persona_map[i] = "💎 VIP Champions"
        elif row['Income'] > inc_mean and row['Spending_Score'] <= spend_mean:
            persona_map[i] = "🎯 Untapped Potential"
        elif row['Income'] <= inc_mean and row['Spending_Score'] > spend_mean:
            persona_map[i] = "🔥 Trendsetters"
        elif row['Income'] <= inc_mean and row['Spending_Score'] <= spend_mean:
            persona_map[i] = "📉 Budget Conscious"
        else:
            persona_map[i] = "⭐ Mainstream"
            
    filtered_df['Persona'] = filtered_df['Cluster'].map(persona_map)

# ==========================================
# 5. MAIN DASHBOARD LAYOUT
# ==========================================
st.markdown("""
    <div class="main-header">
        <h1>📊 Executive Customer Intelligence</h1>
        <p>AI-Driven Customer Segmentation & Behavioral Analytics Platform</p>
    </div>
""", unsafe_allow_html=True)

if filtered_df.empty:
    st.warning("Please select at least one gender from the sidebar.")
    st.stop()

# --- TOP KPI METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers Selected", len(filtered_df))
col2.metric("Average Income", f"${filtered_df['Income'].mean():.1f}k")
col3.metric("Average Spending Score", f"{filtered_df['Spending_Score'].mean():.1f}")
col4.metric("Active Personas Discovered", filtered_df['Persona'].nunique())

st.markdown("<br>", unsafe_allow_html=True)

# --- TABS FOR ORGANIZED ANALYSIS ---
tab1, tab2, tab3 = st.tabs(["🎯 Persona Mapping", "🕸️ Deep Demographics", "🚀 Marketing Strategy"])

# ----- TAB 1: PERSONA MAPPING -----
with tab1:
    st.markdown("### Behavioral Mapping Matrix")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        # Advanced Scatter Plot with Marginal Histograms
        fig_scatter = px.scatter(
            filtered_df, x="Income", y="Spending_Score", color="Persona", size="Age",
            hover_data=["Gender", "Age"],
            marginal_x="box", marginal_y="histogram",
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="Customer Wealth vs. Spending Propensity"
        )
        fig_scatter.update_layout(template="plotly_white", margin=dict(t=50, l=0, r=0, b=0))
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with c2:
        # Advanced Sunburst Chart
        fig_sunburst = px.sunburst(
            filtered_df, path=['Persona', 'Gender'],
            title="Demographic Breakdown by Persona",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_sunburst.update_layout(template="plotly_white", margin=dict(t=50, l=0, r=0, b=0))
        st.plotly_chart(fig_sunburst, use_container_width=True)

# ----- TAB 2: DEEP DEMOGRAPHICS -----
with tab2:
    st.markdown("### Persona Characteristic Breakdown")
    c3, c4 = st.columns(2)
    
    with c3:
        # Calculate Averages for Radar Chart
        radar_df = filtered_df.groupby('Persona')[['Age', 'Income', 'Spending_Score']].mean().reset_index()
        
        # Radar Chart (Spider Web)
        fig_radar = go.Figure()
        for i, row in radar_df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Age']/100, row['Income']/150, row['Spending_Score']/100], # Normalized for scaling
                theta=['Age (Normalized)', 'Income (Normalized)', 'Spending (Normalized)'],
                fill='toself',
                name=row['Persona']
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True, template="plotly_white", title="Persona Attribute Radar"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
    with c4:
        # 3D Visualization
        fig_3d = px.scatter_3d(
            filtered_df, x='Age', y='Income', z='Spending_Score',
            color='Persona', size_max=18, opacity=0.8,
            title="3D Customer Topography"
        )
        fig_3d.update_layout(template="plotly_white", margin=dict(t=50, l=0, r=0, b=0))
        st.plotly_chart(fig_3d, use_container_width=True)

# ----- TAB 3: MARKETING STRATEGY -----
with tab3:
    st.markdown("### 🚀 Actionable Marketing Strategies Based on AI Segments")
    st.markdown("These insights are generated by analyzing the center-points (centroids) of our machine learning clusters.")
    
    # Iterate through unique personas discovered and write business logic
    st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
    for persona in filtered_df['Persona'].unique():
        if "VIP" in persona:
            st.success(f"**{persona}**: These are your most valuable customers. \n* **Action:** Enroll them in an exclusive loyalty program. Send them early-access invitations to premium products.")
        elif "Untapped" in persona:
            st.info(f"**{persona}**: They have high income but don't spend it here. \n* **Action:** Send personalized, targeted advertisements. They need a reason to trust your brand over competitors.")
        elif "Trendsetters" in persona:
            st.warning(f"**{persona}**: They don't make much, but they love spending. \n* **Action:** Push trendy, highly visible, and heavily marketed items. Use social media influencer marketing.")
        elif "Budget" in persona:
            st.error(f"**{persona}**: Low income and low spending. \n* **Action:** Send them extreme discount coupons, clearance sales, and BOGO (Buy One Get One) offers.")
        else:
            st.secondary(f"**{persona}**: The standard mass-market consumer. \n* **Action:** Keep them engaged with standard seasonal newsletters and generalized store promotions.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Show Raw Data for Transparency
    with st.expander("View Raw Clustered Data"):
        st.dataframe(filtered_df.sort_values(by="Persona"), use_container_width=True)
