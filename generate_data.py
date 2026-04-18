"""
Telecom Customer & Network Intelligence Platform
Data Generation & Preprocessing Module
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def generate_telecom_dataset(n_customers=2000):
    """Generate a realistic synthetic telecom dataset."""

    # --- Customer Demographics ---
    customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(1, n_customers + 1)]
    tenure_months = np.random.exponential(scale=24, size=n_customers).clip(1, 72).astype(int)
    age = np.random.normal(38, 12, n_customers).clip(18, 75).astype(int)
    gender = np.random.choice(['Male', 'Female'], n_customers)

    # --- Contract & Billing ---
    contract_type = np.random.choice(
        ['Month-to-month', 'One year', 'Two year'],
        n_customers, p=[0.55, 0.25, 0.20]
    )
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
        n_customers, p=[0.35, 0.20, 0.25, 0.20]
    )
    monthly_charges = np.random.normal(65, 30, n_customers).clip(20, 150)
    total_charges = monthly_charges * tenure_months * np.random.uniform(0.85, 1.05, n_customers)

    # --- Usage Patterns ---
    data_usage_gb = np.random.exponential(scale=8, size=n_customers).clip(0.1, 50)
    voice_minutes = np.random.normal(200, 120, n_customers).clip(0, 800)
    sms_count = np.random.exponential(scale=50, size=n_customers).clip(0, 500).astype(int)
    recharge_frequency = np.random.choice(
        ['Daily', 'Weekly', 'Monthly', 'Rarely'],
        n_customers, p=[0.15, 0.35, 0.40, 0.10]
    )

    # --- Service Subscriptions ---
    has_internet = np.random.choice([1, 0], n_customers, p=[0.75, 0.25])
    has_streaming = np.random.choice([1, 0], n_customers, p=[0.45, 0.55])
    has_mobile_money = np.random.choice([1, 0], n_customers, p=[0.60, 0.40])
    has_roaming = np.random.choice([1, 0], n_customers, p=[0.20, 0.80])
    tech_support = np.random.choice([1, 0], n_customers, p=[0.35, 0.65])

    # --- Support & Complaints ---
    complaint_count = np.random.poisson(lam=1.5, size=n_customers).clip(0, 10)
    avg_support_response_hrs = np.random.exponential(scale=12, size=n_customers).clip(1, 72)

    # --- Derived Features ---
    usage_drop_pct = np.random.uniform(-30, 50, n_customers)
    customer_activity_score = (
        (data_usage_gb / 50) * 30 +
        (voice_minutes / 800) * 25 +
        (sms_count / 500) * 10 +
        has_internet * 15 +
        has_mobile_money * 10 +
        has_streaming * 10
    ).clip(0, 100)

    # --- Network Data ---
    region = np.random.choice(
        ['North', 'South', 'East', 'West', 'Central'],
        n_customers, p=[0.18, 0.22, 0.20, 0.17, 0.23]
    )
    network_type = np.random.choice(['4G', '3G', '2G'], n_customers, p=[0.60, 0.30, 0.10])
    signal_strength = np.random.normal(70, 15, n_customers).clip(20, 100)

    # --- Churn Label (engineered with realistic logic) ---
    churn_score = (
        (contract_type == 'Month-to-month').astype(int) * 0.30 +
        (tenure_months < 12).astype(int) * 0.20 +
        (complaint_count > 3).astype(int) * 0.20 +
        (usage_drop_pct > 20).astype(int) * 0.10 +
        (monthly_charges > 100).astype(int) * 0.10 +
        np.random.uniform(0, 0.10, n_customers)
    )
    churn = (churn_score > 0.45).astype(int)

    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': age,
        'gender': gender,
        'tenure_months': tenure_months,
        'contract_type': contract_type,
        'payment_method': payment_method,
        'monthly_charges': monthly_charges.round(2),
        'total_charges': total_charges.round(2),
        'data_usage_gb': data_usage_gb.round(2),
        'voice_minutes': voice_minutes.round(0).astype(int),
        'sms_count': sms_count,
        'recharge_frequency': recharge_frequency,
        'has_internet': has_internet,
        'has_streaming': has_streaming,
        'has_mobile_money': has_mobile_money,
        'has_roaming': has_roaming,
        'tech_support': tech_support,
        'complaint_count': complaint_count,
        'avg_support_response_hrs': avg_support_response_hrs.round(1),
        'usage_drop_pct': usage_drop_pct.round(1),
        'customer_activity_score': customer_activity_score.round(1),
        'region': region,
        'network_type': network_type,
        'signal_strength': signal_strength.round(1),
        'churn': churn
    })

    return df


def generate_network_data(n_records=5000):
    """Generate synthetic network traffic/congestion data."""
    hours = np.random.choice(range(24), n_records)
    regions = np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_records)
    days = np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], n_records)

    # Peak hours: 7-9am and 6-9pm
    is_peak = ((hours >= 7) & (hours <= 9)) | ((hours >= 18) & (hours <= 21))
    base_traffic = np.random.exponential(scale=200, size=n_records)
    traffic_mbps = np.where(is_peak, base_traffic * 2.5, base_traffic).clip(10, 2000)
    congestion_level = (traffic_mbps / 2000 * 100 + np.random.normal(0, 5, n_records)).clip(0, 100)
    latency_ms = (20 + congestion_level * 0.8 + np.random.normal(0, 5, n_records)).clip(5, 200)
    packet_loss_pct = (congestion_level * 0.05 + np.random.exponential(0.5, n_records)).clip(0, 15)
    active_users = (traffic_mbps / 5 + np.random.normal(0, 10, n_records)).clip(1, 500).astype(int)

    return pd.DataFrame({
        'hour': hours,
        'day': days,
        'region': regions,
        'traffic_mbps': traffic_mbps.round(1),
        'congestion_level': congestion_level.round(1),
        'latency_ms': latency_ms.round(1),
        'packet_loss_pct': packet_loss_pct.round(2),
        'active_users': active_users
    })


def preprocess_data(df):
    """Preprocess customer dataframe for ML."""
    df_proc = df.copy()

    # Encode categoricals
    le = LabelEncoder()
    cat_cols = ['gender', 'contract_type', 'payment_method', 'recharge_frequency',
                'region', 'network_type']
    for col in cat_cols:
        df_proc[col + '_enc'] = le.fit_transform(df_proc[col])

    # Feature matrix
    feature_cols = [
        'age', 'tenure_months', 'monthly_charges', 'total_charges',
        'data_usage_gb', 'voice_minutes', 'sms_count',
        'has_internet', 'has_streaming', 'has_mobile_money', 'has_roaming', 'tech_support',
        'complaint_count', 'avg_support_response_hrs',
        'usage_drop_pct', 'customer_activity_score', 'signal_strength',
        'gender_enc', 'contract_type_enc', 'payment_method_enc',
        'recharge_frequency_enc', 'region_enc', 'network_type_enc'
    ]

    X = df_proc[feature_cols]
    y = df_proc['churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_cols, scaler, df_proc


if __name__ == "__main__":
    df = generate_telecom_dataset()
    net_df = generate_network_data()
    print(f"Customer dataset: {df.shape}")
    print(f"Network dataset: {net_df.shape}")
    print(f"Churn rate: {df['churn'].mean():.2%}")
    print(df.head())
