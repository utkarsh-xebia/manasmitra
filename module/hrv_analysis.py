"""
Heart Rate Variability (HRV) Analysis and Stress Classification
This script analyzes heart rate data, calculates HRV metrics, and trains ML models for stress classification.
"""

import pandas as pd
import numpy as np
import datetime
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Try importing XGBoost if available
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')

# ============================================================================
# 1. Understand the Dataset
# ============================================================================
print("=" * 80)
print("1. UNDERSTANDING THE DATASET")
print("=" * 80)

# Load the dataset
df = pd.read_csv('heart_rate.csv')

print(f"\nDataset Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn Information:")
print(df.info())

# Identify timestamp column
# Analysis shows columns T1, T2, T3, T4 are heart rate values.
# No explicit timestamp column found. We will generate one assuming 1Hz frequency.
print("\nTimestamp Analysis:")
print("No explicit timestamp column found in the dataset.")
print("Columns T1, T2, T3, T4 represent heart rate (BPM) from different sessions/subjects.")

# Check for missing values
print("\nMissing Values per Column:")
print(df.isnull().sum())

# ============================================================================
# 2. Preprocessing
# ============================================================================
print("\n" + "=" * 80)
print("2. PREPROCESSING")
print("=" * 80)

# Since we have 4 different heart rate series, let's focus on T1 for the main analysis
# but we can process all of them. Let's melt the dataframe to have a long format.
df_melted = df.melt(var_name='Session', value_name='HeartRate')
df_melted = df_melted.dropna().reset_index(drop=True)

# Generate timestamps
# Assuming 1Hz frequency (1 reading per second) starting from a reference time
start_time = datetime.datetime(2026, 1, 11, 8, 0, 0)
df_melted['Timestamp'] = [start_time + datetime.timedelta(seconds=i) for i in range(len(df_melted))]

# Convert timestamp to proper datetime (already done during generation)
df_melted['Timestamp'] = pd.to_datetime(df_melted['Timestamp'])

# Sort data by time
df_melted = df_melted.sort_values('Timestamp')

# Remove noise & outliers (Heart rate should be between 40 and 200 BPM)
df_clean = df_melted[(df_melted['HeartRate'] >= 40) & (df_melted['HeartRate'] <= 200)].copy()

# Handle missing heartbeats - Interpolate if there are gaps (not applicable here as we generated time)
# But let's check for any jumps in values
df_clean['HeartRate'] = df_clean['HeartRate'].interpolate()

print(f"Data after cleaning and timestamp generation: {df_clean.shape}")
print(df_clean.head())

# ============================================================================
# 3. HRV Calculation (Main Part)
# ============================================================================
print("\n" + "=" * 80)
print("3. HRV CALCULATION")
print("=" * 80)

def calculate_hrv_metrics(hr_series):
    # Calculate RR intervals in milliseconds
    # RR = 60000 / HR
    rr_intervals = 60000 / hr_series
    
    # SDNN: Standard deviation of RR intervals
    sdnn = np.std(rr_intervals)
    
    # RMSSD: Root Mean Square of Successive Differences
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr**2))
    
    # pNN50: Percentage of successive RR intervals that differ by more than 50ms
    nn50 = np.sum(np.abs(diff_rr) > 50)
    pnn50 = (nn50 / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0
    
    # Mean RR
    mean_rr = np.mean(rr_intervals)
    
    return {
        'SDNN': sdnn,
        'RMSSD': rmssd,
        'pNN50': pnn50,
        'MeanRR': mean_rr
    }

# Calculate global metrics
global_metrics = calculate_hrv_metrics(df_clean['HeartRate'])

print("\nGlobal HRV Metrics:")
for metric, value in global_metrics.items():
    print(f"  {metric}: {value:.4f}")

print("\nMetric Explanations:")
print("- SDNN: Reflects overall heart rate variability. Higher values usually indicate better health.")
print("- RMSSD: Reflects parasympathetic activity (relaxation). Primary metric for short-term stress.")
print("- pNN50: Percentage of heartbeats with >50ms difference. Indicates high-frequency variability.")
print("- MeanRR: Average time between heartbeats. Inverse of average heart rate.")

# ============================================================================
# 4. Feature Engineering
# ============================================================================
print("\n" + "=" * 80)
print("4. FEATURE ENGINEERING")
print("=" * 80)

# Create rolling HRV (window of 60 samples/seconds)
window_size = 60
df_clean['RR'] = 60000 / df_clean['HeartRate']

# SDNN Rolling
df_clean['Rolling_SDNN'] = df_clean['RR'].rolling(window=window_size).std()

# RMSSD Rolling
def calc_rmssd_rolling(x):
    if len(x) < 2: return np.nan
    return np.sqrt(np.mean(np.diff(x)**2))

df_clean['Rolling_RMSSD'] = df_clean['RR'].rolling(window=window_size).apply(calc_rmssd_rolling)

# Resting HRV (Median HRV)
resting_hrv = df_clean['Rolling_RMSSD'].median()

# Stress Score (Inverse of RMSSD normalized)
# Simple stress score: 100 - (normalized RMSSD)
max_rmssd = df_clean['Rolling_RMSSD'].max()
min_rmssd = df_clean['Rolling_RMSSD'].min()
df_clean['Stress_Score'] = 100 * (1 - (df_clean['Rolling_RMSSD'] - min_rmssd) / (max_rmssd - min_rmssd))

# Drop NaNs created by rolling windows
df_features = df_clean.dropna().copy()

print("\nFeature Engineering Complete. New features added:")
print("- Rolling_SDNN, Rolling_RMSSD, Stress_Score")

# ============================================================================
# 5. Stress Classification
# ============================================================================
print("\n" + "=" * 80)
print("5. STRESS CLASSIFICATION")
print("=" * 80)

# Create labels based on RMSSD (Standard approach: Low RMSSD = High Stress)
# Using quantiles for relative labeling
q1 = df_features['Rolling_RMSSD'].quantile(0.33)
q2 = df_features['Rolling_RMSSD'].quantile(0.66)

def classify_stress(rmssd):
    if rmssd < q1:
        return 'Stressed'
    elif rmssd < q2:
        return 'Normal'
    else:
        return 'Relaxed'

df_features['Stress_Label'] = df_features['Rolling_RMSSD'].apply(classify_stress)

# Add emojis to labels
emoji_map = {'Relaxed': 'Relaxed', 'Normal': 'Normal', 'Stressed': 'Stressed'}
df_features['Stress_Emoji'] = df_features['Stress_Label'].map(emoji_map)

print("\nStress Label Distribution:")
print(df_features['Stress_Emoji'].value_counts())

# ============================================================================
# 6. Model Building
# ============================================================================
print("\n" + "=" * 80)
print("6. MODEL BUILDING")
print("=" * 80)

# Features for ML
X = df_features[['HeartRate', 'Rolling_SDNN', 'Rolling_RMSSD', 'Stress_Score']]
y = df_features['Stress_Label']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary to store results
results = {}

# 1. Logistic Regression
print("\nTraining Logistic Regression...")
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
results['Logistic Regression'] = accuracy_score(y_test, y_pred_lr)

# 2. Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results['Random Forest'] = accuracy_score(y_test, y_pred_rf)

# 3. XGBoost
if XGBOOST_AVAILABLE:
    print("Training XGBoost...")
    xgb = XGBClassifier(eval_metric='mlogloss')
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    results['XGBoost'] = accuracy_score(y_test, y_pred_xgb)

print("\nModel Comparison (Accuracy):")
for model, acc in results.items():
    print(f"  {model}: {acc:.4f}")

# Select best model
best_model_name = max(results, key=results.get)
print(f"\nBest Model: {best_model_name}")

# Detailed report for best model
if best_model_name == 'Logistic Regression':
    final_model = lr
    final_pred = y_pred_lr
elif best_model_name == 'Random Forest':
    final_model = rf
    final_pred = y_pred_rf
else:
    final_model = xgb
    final_pred = y_pred_xgb

print("\nClassification Report (Best Model):")
print(classification_report(y_test, final_pred, target_names=le.classes_))

# ============================================================================
# 7. Save Model
# ============================================================================
print("\n" + "=" * 80)
print("7. SAVING MODEL")
print("=" * 80)

model_package = {
    'model': final_model,
    'scaler': scaler if best_model_name == 'Logistic Regression' else None,
    'label_encoder': le,
    'best_model_name': best_model_name,
    'features': list(X.columns)
}

joblib.dump(model_package, 'hrv_stress_model.joblib')
print("Model and scaler saved to 'hrv_stress_model.joblib'")

print("\nReloading Code Example:")
print("-" * 40)
print("""
import joblib
import pandas as pd

# Load the package
package = joblib.load('hrv_stress_model.joblib')
model = package['model']
scaler = package['scaler']
le = package['label_encoder']

# Example prediction
# new_data = pd.DataFrame(...) 
# if scaler: new_data = scaler.transform(new_data)
# prediction = model.predict(new_data)
# label = le.inverse_transform(prediction)
""")
print("-" * 40)

print("\nHRV Analysis and Stress Classification Complete!")
