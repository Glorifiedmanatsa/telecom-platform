"""
Telecom Customer & Network Intelligence Platform
Machine Learning Models Module
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# CHURN PREDICTION
# ─────────────────────────────────────────────

def train_churn_model(X, y, feature_cols):
    """Train and evaluate churn prediction models."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            class_weight='balanced', random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        )
    }

    results = {}
    best_model = None
    best_auc = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc_roc': auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

        if auc > best_auc:
            best_auc = auc
            best_model = model

    # Feature importances (from RF)
    rf = results['Random Forest']['model']
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    return results, best_model, importances, (X_train, X_test, y_train, y_test)


def predict_churn_single(model, scaler, customer_features: dict, feature_cols: list):
    """Predict churn for a single customer."""
    row = pd.DataFrame([customer_features])
    # Fill missing cols with 0
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_cols]
    row_scaled = scaler.transform(row)
    prob = model.predict_proba(row_scaled)[0][1]
    risk = 'High' if prob > 0.65 else ('Medium' if prob > 0.35 else 'Low')
    return prob, risk


# ─────────────────────────────────────────────
# CUSTOMER SEGMENTATION
# ─────────────────────────────────────────────

def train_segmentation_model(X, df_raw, n_clusters=4):
    """Cluster customers into meaningful segments."""
    # Find optimal k using silhouette
    sil_scores = {}
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil_scores[k] = silhouette_score(X, labels)

    # Use provided n_clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    sil = silhouette_score(X, cluster_labels)

    # Annotate raw df with clusters
    df_seg = df_raw.copy()
    df_seg['cluster'] = cluster_labels

    # Profile each cluster
    profile_cols = ['data_usage_gb', 'voice_minutes', 'monthly_charges',
                    'customer_activity_score', 'churn', 'tenure_months', 'complaint_count']
    cluster_profiles = df_seg.groupby('cluster')[profile_cols].mean().round(2)

    # Assign segment names based on activity score & charges
    seg_names = {}
    ranked = cluster_profiles['customer_activity_score'].rank(ascending=False)
    for cluster_id in cluster_profiles.index:
        rank = ranked[cluster_id]
        charges = cluster_profiles.loc[cluster_id, 'monthly_charges']
        if rank == 1:
            seg_names[cluster_id] = '⭐ High-Value Users'
        elif rank == 2 and charges > 60:
            seg_names[cluster_id] = '📊 Moderate-Active Users'
        elif rank == 3:
            seg_names[cluster_id] = '📉 Low-Activity Users'
        else:
            seg_names[cluster_id] = '💤 At-Risk / Churners'

    df_seg['segment_name'] = df_seg['cluster'].map(seg_names)

    return kmeans, df_seg, cluster_profiles, sil_scores, sil, seg_names


# ─────────────────────────────────────────────
# BUNDLE RECOMMENDATION
# ─────────────────────────────────────────────

BUNDLE_CATALOG = {
    'Basic Voice': {
        'price': 5, 'data_gb': 0, 'minutes': 200, 'sms': 100,
        'description': 'Voice-only starter plan'
    },
    'Data Starter': {
        'price': 10, 'data_gb': 2, 'minutes': 50, 'sms': 50,
        'description': 'Light data for casual users'
    },
    'Smart Bundle S': {
        'price': 20, 'data_gb': 5, 'minutes': 150, 'sms': 200,
        'description': 'Balanced plan for everyday use'
    },
    'Smart Bundle M': {
        'price': 35, 'data_gb': 15, 'minutes': 300, 'sms': 500,
        'description': 'Popular mid-tier plan'
    },
    'Power User': {
        'price': 60, 'data_gb': 40, 'minutes': 600, 'sms': 1000,
        'description': 'Heavy data & calls'
    },
    'Unlimited Pro': {
        'price': 90, 'data_gb': 100, 'minutes': 1000, 'sms': 2000,
        'description': 'Unlimited everything for professionals'
    },
    'Mobile Money+': {
        'price': 25, 'data_gb': 8, 'minutes': 200, 'sms': 300,
        'description': 'Bundle with mobile money benefits'
    },
    'Streaming Pack': {
        'price': 45, 'data_gb': 30, 'minutes': 200, 'sms': 100,
        'description': 'Optimized for video streaming'
    },
}


def recommend_bundle(customer: dict) -> list:
    """Rule-based + score-matching bundle recommendation."""
    scores = {}
    data_gb = customer.get('data_usage_gb', 5)
    minutes = customer.get('voice_minutes', 200)
    has_mobile_money = customer.get('has_mobile_money', 0)
    has_streaming = customer.get('has_streaming', 0)
    budget = customer.get('monthly_charges', 50)
    activity = customer.get('customer_activity_score', 50)

    for bundle_name, bundle in BUNDLE_CATALOG.items():
        score = 0
        # Data match
        data_diff = abs(bundle['data_gb'] - data_gb)
        score += max(0, 30 - data_diff * 2)
        # Minutes match
        min_diff = abs(bundle['minutes'] - minutes)
        score += max(0, 20 - min_diff * 0.05)
        # Budget proximity
        price_diff = abs(bundle['price'] - budget * 0.5)
        score += max(0, 20 - price_diff)
        # Special affinity
        if has_mobile_money and bundle_name == 'Mobile Money+':
            score += 15
        if has_streaming and bundle_name == 'Streaming Pack':
            score += 15
        if activity > 70 and bundle['data_gb'] >= 30:
            score += 10
        scores[bundle_name] = round(score, 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    recommendations = []
    for name, score in ranked[:3]:
        bundle_info = BUNDLE_CATALOG[name].copy()
        bundle_info['name'] = name
        bundle_info['score'] = score
        recommendations.append(bundle_info)
    return recommendations


# ─────────────────────────────────────────────
# NETWORK CONGESTION
# ─────────────────────────────────────────────

def analyze_network(net_df: pd.DataFrame) -> dict:
    """Analyze network traffic and congestion patterns."""
    # Average congestion by hour
    hourly = net_df.groupby('hour').agg({
        'traffic_mbps': 'mean',
        'congestion_level': 'mean',
        'latency_ms': 'mean',
        'active_users': 'mean'
    }).round(2)

    # By region
    regional = net_df.groupby('region').agg({
        'traffic_mbps': 'mean',
        'congestion_level': 'mean',
        'latency_ms': 'mean',
        'packet_loss_pct': 'mean'
    }).round(2)

    # By day
    day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    daily = net_df.groupby('day')['congestion_level'].mean().reindex(day_order).round(2)

    # Peak hours
    peak_hours = hourly['congestion_level'].nlargest(3).index.tolist()

    # High-congestion zones
    high_congestion_regions = regional[regional['congestion_level'] > 60].index.tolist()

    return {
        'hourly': hourly,
        'regional': regional,
        'daily': daily,
        'peak_hours': peak_hours,
        'high_congestion_regions': high_congestion_regions,
        'avg_congestion': net_df['congestion_level'].mean().round(1),
        'avg_latency': net_df['latency_ms'].mean().round(1),
        'avg_packet_loss': net_df['packet_loss_pct'].mean().round(2)
    }


if __name__ == "__main__":
    from data.generate_data import generate_telecom_dataset, generate_network_data, preprocess_data
    df = generate_telecom_dataset()
    net_df = generate_network_data()
    X, y, feature_cols, scaler, df_proc = preprocess_data(df)

    print("Training churn model...")
    results, best_model, importances, splits = train_churn_model(X, y, feature_cols)
    for name, r in results.items():
        print(f"{name}: AUC={r['auc_roc']:.3f}, F1={r['f1']:.3f}")

    print("\nTraining segmentation model...")
    kmeans, df_seg, profiles, sil_scores, sil, seg_names = train_segmentation_model(X, df)
    print(f"Silhouette score: {sil:.3f}")
    print(profiles)

    print("\nNetwork analysis...")
    net_analysis = analyze_network(net_df)
    print(f"Peak hours: {net_analysis['peak_hours']}")
    print(f"Avg congestion: {net_analysis['avg_congestion']}%")
