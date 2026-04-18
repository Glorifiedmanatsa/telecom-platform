# 📡 Telecom Customer & Network Intelligence Platform

A machine learning-powered analytics dashboard for telecom customer intelligence and network monitoring.

## 🎯 Features

| Module | Description |
|---|---|
| 🔮 Churn Prediction | Random Forest, Gradient Boosting & Logistic Regression models |
| 👥 Customer Segmentation | K-Means clustering with 4 behavioral segments |
| 📦 Bundle Recommender | Usage-based personalized bundle suggestions |
| 🌐 Network Analytics | Hourly/regional congestion & latency analysis |
| 🔍 Customer Lookup | Individual customer risk profile & recommendations |

---

## 🚀 Local Setup & Run

### 1. Clone / Download the project
```bash
# If using git:
git clone <your-repo-url>
cd telecom_platform

# Or unzip the folder and cd into it
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv

# Activate:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## ☁️ Deploy to Streamlit Cloud (Free)

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial telecom platform"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/telecom-platform.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository, branch (`main`), and set **Main file path** to `app.py`
5. Click **Deploy** — your app will be live in ~2 minutes!

### Free URL format:
```
https://YOUR_USERNAME-telecom-platform-app-XXXX.streamlit.app
```

---

## 🗂️ Project Structure

```
telecom_platform/
├── app.py                    ← Main Streamlit dashboard (entry point)
├── requirements.txt          ← Python dependencies
├── README.md
├── data/
│   ├── __init__.py
│   └── generate_data.py      ← Synthetic data generation + preprocessing
└── models/
    ├── __init__.py
    └── ml_models.py          ← Churn, segmentation, recommendation, network models
```

---

## 🤖 ML Models Used

### Churn Prediction (Classification)
- **Random Forest** — Best performer, provides feature importances
- **Gradient Boosting** — High AUC-ROC
- **Logistic Regression** — Baseline comparison
- Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC

### Customer Segmentation (Clustering)
- **K-Means** with k=4 clusters
- Evaluated with Silhouette Score
- Segments: High-Value, Moderate-Active, Low-Activity, At-Risk

### Bundle Recommendation
- Rule-based scoring system matching usage patterns to bundle catalog
- Factors: data usage, voice minutes, budget, special services

### Network Analytics
- Hourly & regional traffic aggregation
- Congestion threshold detection
- Latency & packet loss correlation analysis

---

## 📊 Dataset

Based on **IBM Telco Customer Churn** dataset (public) with additional synthetic features:
- 2,000 customer records
- 5,000 network traffic records
- 25 customer features + network metrics

---

## 🔮 Future Enhancements (Chapter 6)

- [ ] Real-time data streaming integration
- [ ] Deep learning models (LSTM for usage forecasting)
- [ ] Geographic network mapping (Folium/Mapbox)
- [ ] Mobile money analytics integration
- [ ] CRM system API integration
- [ ] Automated alert notifications

---

## 👨‍💻 Tech Stack

| Tool | Use |
|---|---|
| Python 3.10+ | Core language |
| Streamlit | Web dashboard |
| Scikit-learn | ML models |
| Pandas / NumPy | Data processing |
| Plotly | Interactive charts |
| XGBoost | Gradient boosting |

---

*Department of Computer Science & Engineering, PIT Vadodara*
