import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(page_title="Customer Segmentation Pro", page_icon="🛍️", layout="wide")

# Custom CSS for a cleaner, modern look
st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #f1f3f6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Load the Data
@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    return df

df = load_data()

# 3. Sidebar Navigation & Controls
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3126/3126647.png", width=100)
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", ["📊 Data Overview", "🧠 K-Means Clustering", "📈 Cluster Insights"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Built with 💻 Streamlit**")

# ==========================================
# PAGE 1: DATA OVERVIEW
# ==========================================
if menu == "📊 Data Overview":
    st.title("📊 Customer Demographics & Overview")
    st.markdown("Explore the raw dataset and understand the distribution of our mall customers before applying machine learning.")
    
    # Top KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", len(df))
    col2.metric("Average Age", f"{df['Age'].mean():.0f} yrs")
    col3.metric("Average Income", f"${df['Annual Income (k$)'].mean():.1f}k")
    col4.metric("Avg Spending Score", f"{df['Spending Score (1-100)'].mean():.1f}")
    
    st.markdown("---")
    
    # Visualizations
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Gender Distribution")
        fig_gender = px.pie(df, names='Gender', hole=0.4, color_discrete_sequence=['#ff9999','#66b3ff'])
        fig_gender.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_gender, use_container_width=True)
        
    with c2:
        st.subheader("Income vs Spending Score")
        fig_scatter = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)', color='Gender', opacity=0.7)
        fig_scatter.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    st.subheader("Raw Dataset")
    st.dataframe(df, use_container_width=True)

# ==========================================
# PAGE 2: K-MEANS CLUSTERING
# ==========================================
elif menu == "🧠 K-Means Clustering":
    st.title("🧠 K-Means Clustering Model")
    st.markdown("Use the slider below to adjust the number of clusters (K) and see how the machine learning algorithm groups customers.")
    
    # Model Controls
    n_clusters = st.slider("Select Number of Clusters (K):", min_value=2, max_value=10, value=5)
    
    # Data Prep & Modeling
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X).astype(str) # String for categorical colors
    
    st.markdown("---")
    
    # Interactive Plot
    st.subheader(f"Customer Segments (K={n_clusters})")
    fig_cluster = px.scatter(
        df, 
        x='Annual Income (k$)', 
        y='Spending Score (1-100)', 
        color='Cluster',
        size='Age',
        hover_data=['Gender', 'Age'],
        color_discrete_sequence=px.colors.qualitative.Bold,
        title="Hover over the bubbles to see customer details!"
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

# ==========================================
# PAGE 3: CLUSTER INSIGHTS
# ==========================================
elif menu == "📈 Cluster Insights":
    st.title("📈 Actionable Business Insights")
    st.markdown("Breakdown of the customer segments to help drive marketing decisions.")
    
    # We must compute the clusters again for this page
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    
    # Group data by cluster
    cluster_summary = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().round(1).reset_index()
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Cluster Averages")
        st.dataframe(cluster_summary, hide_index=True)
        
    with c2:
        st.subheader("Marketing Strategy Recommendations")
        st.success("**Target Premium (High Income, High Spend):** Send exclusive VIP invites and showcase luxury products.")
        st.warning("**Target Potential (High Income, Low Spend):** Send highly targeted ads and discounts to win their loyalty.")
        st.info("**Target Mass Market (Average Income, Average Spend):** Focus on general campaigns and volume sales.")
        st.error("**Target Careful (Low Income, Low Spend):** Highlight budget-friendly options and heavy discounts.")
