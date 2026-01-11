"""
Fusion-based Depression Severity Prediction System
Combines PHQ-9 questionnaire data and Heart Rate LSTM embeddings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

warnings.filterwarnings('ignore')

# ============================================================================
# 1. Load and Merge Datasets
# ============================================================================
print("=" * 80)
print("1. LOADING AND MERGING DATASETS")
print("=" * 80)

# Load datasets
df_phq = pd.read_csv('PHQ-9_Dataset_5th Edition.csv')
df_hr = pd.read_csv('heart_rate.csv')

# Data cleaning for PHQ
df_phq.columns = df_phq.columns.str.strip()
text_columns = df_phq.select_dtypes(include=['object']).columns
for col in text_columns:
    df_phq[col] = df_phq[col].astype(str).str.strip().str.lower()

# Since we don't have a common ID, merge row-wise (truncate to smaller size)
min_samples = min(len(df_phq), len(df_hr))
df_combined = pd.concat([df_phq.iloc[:min_samples].reset_index(drop=True), 
                        df_hr[['T1', 'T2', 'T3', 'T4']].iloc[:min_samples].reset_index(drop=True)], axis=1)

# Handle missing values in HR columns
df_combined[['T1', 'T2', 'T3', 'T4']] = df_combined[['T1', 'T2', 'T3', 'T4']].fillna(method='ffill').fillna(method='bfill')

print(f"Combined dataset size: {df_combined.shape}")

# ============================================================================
# 2. Preprocessing
# ============================================================================
print("\n" + "=" * 80)
print("2. PREPROCESSING")
print("=" * 80)

# PHQ-9 Encoding
phq_encoding = {'not at all': 0, 'several days': 1, 'more than half the days': 2, 'nearly every day': 3}
# Identify PHQ columns (first 9 questions)
non_phq_cols = ['Age', 'Gender', 'PHQ_Total', 'PHQ_Severity', 'Sleep Quality', 'Study Pressure', 'Financial Pressure', 'T1', 'T2', 'T3', 'T4']
phq_question_cols = [col for col in df_combined.columns if col not in non_phq_cols]

for col in phq_question_cols:
    df_combined[col] = df_combined[col].map(phq_encoding).fillna(0).astype(int)

# Categorical Encoding
le_gender = LabelEncoder()
df_combined['Gender_encoded'] = le_gender.fit_transform(df_combined['Gender'])

mapping = {'good': 0, 'average': 1, 'bad': 2, 'worst': 3}
df_combined['Sleep_encoded'] = df_combined['Sleep Quality'].map(mapping).fillna(1).astype(int)
df_combined['Study_encoded'] = df_combined['Study Pressure'].map(mapping).fillna(1).astype(int)
df_combined['Financial_encoded'] = df_combined['Financial Pressure'].map(mapping).fillna(1).astype(int)

# Severity Label Encoding
le_severity = LabelEncoder()
df_combined['Target'] = le_severity.fit_transform(df_combined['PHQ_Severity'])

# Select PHQ features
phq_features = phq_question_cols + ['Age', 'Gender_encoded', 'Sleep_encoded', 'Study_encoded', 'Financial_encoded']
X_phq = df_combined[phq_features].values
y = df_combined['Target'].values

# ============================================================================
# 3. HRV Model (LSTM)
# ============================================================================
print("\n" + "=" * 80)
print("3. HRV MODEL (LSTM)")
print("=" * 80)

# Reshape T1-T4 for LSTM: (samples, timesteps, features)
X_hrv = df_combined[['T1', 'T2', 'T3', 'T4']].values
X_hrv_seq = X_hrv.reshape((X_hrv.shape[0], 4, 1))

# Scale HRV data
scaler_hrv = StandardScaler()
X_hrv_seq_scaled = scaler_hrv.fit_transform(X_hrv.reshape(-1, 1)).reshape(X_hrv.shape[0], 4, 1)

# Build LSTM Model
input_layer = Input(shape=(4, 1))
lstm_layer = LSTM(32, activation='tanh', return_sequences=False)(input_layer)
dropout = Dropout(0.2)(lstm_layer)
embedding_layer = Dense(16, activation='relu', name='embedding')(dropout)
output_layer = Dense(len(le_severity.classes_), activation='softmax')(embedding_layer)

lstm_model = Model(inputs=input_layer, outputs=output_layer)
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train LSTM
print("Training LSTM model to learn embeddings...")
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_hrv_seq_scaled, y, test_size=0.2, random_state=42)
lstm_model.fit(X_train_h, y_train_h, epochs=20, batch_size=16, verbose=0, validation_split=0.1)

# Extract Embeddings
embedding_model = Model(inputs=lstm_model.input, outputs=lstm_model.get_layer('embedding').output)
X_embeddings = embedding_model.predict(X_hrv_seq_scaled)

print(f"Embeddings extracted. Shape: {X_embeddings.shape}")

# Save LSTM model
lstm_model.save('hrv_lstm_model.h5')

# ============================================================================
# 4. Fusion & Classification
# ============================================================================
print("\n" + "=" * 80)
print("4. FUSION & CLASSIFICATION")
print("=" * 80)

# Standardize PHQ features and HRV embeddings
scaler_phq = StandardScaler()
X_phq_scaled = scaler_phq.fit_transform(X_phq)

scaler_embed = StandardScaler()
X_embed_scaled = scaler_embed.fit_transform(X_embeddings)

# Split data for fusion classifier
X_train_phq, X_test_phq, X_train_emb, X_test_emb, y_train, y_test = train_test_split(
    X_phq_scaled, X_embed_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Experiment with different fusion weight ratios
# Ratio: w * PHQ + (1-w) * Embeddings
weights = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
results = []

print("\nEvaluating fusion weights...")

for w in weights:
    # Weighted concatenation (multiplying features by weight)
    X_train_fused = np.hstack([X_train_phq * w, X_train_emb * (1 - w)])
    X_test_fused = np.hstack([X_test_phq * w, X_test_emb * (1 - w)])
    
    # Try different classifiers (using Random Forest as representative)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_fused, y_train)
    y_pred = clf.predict(X_test_fused)
    acc = accuracy_score(y_test, y_pred)
    
    results.append({'weight': w, 'accuracy': acc})
    print(f"Weight Ratio {w:.1f} PHQ / {1-w:.1f} HRV -> Accuracy: {acc:.4f}")

# Save the best fusion model (e.g., at weight 0.5)
X_train_fused_best = np.hstack([X_train_phq * 0.5, X_train_emb * 0.5])
best_clf = RandomForestClassifier(n_estimators=100, random_state=42)
best_clf.fit(X_train_fused_best, y_train)
joblib.dump(best_clf, 'fused_depression_model.joblib')

# Save preprocessing objects
preprocessing = {
    'scaler_phq': scaler_phq,
    'scaler_embed': scaler_embed,
    'scaler_hrv': scaler_hrv,
    'le_gender': le_gender,
    'le_severity': le_severity,
    'phq_question_cols': phq_question_cols,
    'phq_features': phq_features
}
joblib.dump(preprocessing, 'fusion_preprocessing.joblib')

# ============================================================================
# 5. Visualization
# ============================================================================
print("\n" + "=" * 80)
print("5. VISUALIZATION")
print("=" * 80)

weights_list = [r['weight'] for r in results]
acc_list = [r['accuracy'] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(weights_list, acc_list, marker='o', linestyle='-', color='blue', linewidth=2)

# Add labels and title
plt.title('Fusion Model Performance vs. PHQ Weight Ratio', fontsize=14)
plt.xlabel('PHQ Feature Weight (1-w is HRV Weight)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Add markers and labels
for i, acc in enumerate(acc_list):
    plt.text(weights_list[i], acc + 0.005, f"{acc:.2%}", ha='center')

plt.ylim(min(acc_list) - 0.05, 1.0)
plt.savefig('fusion_performance_plot.png')
print("Visualization saved as 'fusion_performance_plot.png'")

print("\nAll tasks completed successfully!")
