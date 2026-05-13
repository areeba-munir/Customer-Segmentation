# 📊 Executive Customer Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Data%20Viz-Plotly-3F4F75.svg)](https://plotly.com/)

An AI-driven customer segmentation and behavioral analytics dashboard built to help modern marketing teams transition from mass-marketing to highly targeted, persona-based campaigns. 

**[🔗 Click Here to View the Live Application](Insert_Your_Streamlit_Cloud_Link_Here)**

---

## 💡 The Business Problem
Understanding consumer behavior is critical for maximizing Return on Investment (ROI) in marketing. Without clear segmentation, businesses waste capital advertising premium products to budget-conscious buyers, or offering heavy discounts to consumers who are already willing to pay full price.

## 🚀 The Solution
This web application leverages **Unsupervised Machine Learning (K-Means Clustering)** to automatically identify hidden patterns in customer wealth and spending propensity. It dynamically categorizes the customer base into distinct, actionable marketing personas:

* **High-Value Premium:** High income, high spenders.
* **Conservative Wealth:** High income, low spenders (untapped potential).
* **Aspirational Spenders:** Lower income, high spenders (trendsetters).
* **Value Seekers:** Low income, low spenders (budget-conscious).
* **Core Market:** Average income, average spenders.

---

## ✨ Key Features

* **Dynamic ML Clustering:** Users can adjust the target number of segments (K) on the fly, instantly recalculating the K-Means algorithm and updating the dashboard.
* **Algorithmic Persona Generation:** The application mathematically analyzes the centroids (center points) of the generated clusters against global averages to automatically assign professional industry-standard persona names.
* **Deep Demographics:** Interactive 3D topographies, marginal histograms, and normalized radar (spider) charts built with `Plotly` to visualize multivariate customer traits.
* **Actionable Marketing Strategy:** A dedicated logic engine that translates the mathematical clusters into plain-English, actionable marketing strategies for business stakeholders.

---

## 🛠️ Technology Stack

* **Core Language:** Python
* **Frontend Framework:** Streamlit (Customized with CSS for an executive SaaS UI)
* **Machine Learning:** Scikit-Learn (`KMeans` clustering)
* **Data Manipulation:** Pandas, NumPy
* **Interactive Visualization:** Plotly Express, Plotly Graph Objects

---

## 💻 Local Installation & Usage

If you would like to run this project locally on your own machine, follow these steps:

**1. Clone the repository**
```bash
git clone [https://github.com/YourUsername/Your-Repository-Name.git](https://github.com/YourUsername/Your-Repository-Name.git)
cd Your-Repository-Name
