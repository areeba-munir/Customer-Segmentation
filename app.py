import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# 1. Page Configuration
st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")
st.title("🛍️ Mall Customer Segmentation Dashboard")
st.markdown("This application uses **Machine Learning (K-Means Clustering)** to segment customers based on their purchasing behavior.")

# 2. Load the Data
@st.cache_data
def load_data():
    # Make sure 'Mall_Customers.csv' is in the exact same folder as this script
    df = pd.read_csv("Mall_Customers.csv")
    return df

df = load_data()

# 3. Sidebar for User Inputs
st.sidebar.header("Model Parameters")
st.sidebar.markdown("Choose the number of clusters (segments) you want to divide the customers into.")
# For this dataset, 5 is usually the optimal number of clusters
n_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=5)

# 4. Display Raw Data
with st.expander("Preview the Dataset"):
    st.dataframe(df.head(10))

# 5. Data Preparation and Modeling
# We focus on Income and Spending Score for 2D visualization
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Initialize and fit the K-Means model
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# 6. Data Visualization
st.subheader(f"Customer Segments (K = {n_clusters})")

# Create a Matplotlib figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the clusters using seaborn
sns.scatterplot(
    x='Annual Income (k$)', 
    y='Spending Score (1-100)', 
    hue='Cluster', 
    data=df, 
    palette='Set1', 
    s=100, 
    ax=ax
)

# Plot the centroids (the center of each cluster)
centroids = kmeans.cluster_centers_
ax.scatter(centroids[:, 0], centroids[:, 1], s=300, c='black', marker='X', label='Centroids')

ax.set_title("Clusters of Customers")
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.legend()

# Show the plot in Streamlit
st.pyplot(fig)

# 7. Cluster Insights
st.markdown("### 💡 Cluster Insights")
st.markdown("""
* **High Income / High Spending:** Target these customers for premium products.
* **High Income / Low Spending:** High potential! Try to attract them with targeted marketing.
* **Low Income / High Spending:** Careless spenders, target with trendy items. 
* **Low Income / Low Spending:** Budget-conscious standard customers.
* **Average Income / Average Spending:** The general mass market.
""")