"""
PHQ-9 Mental Health Assessment ML Model
Designed for Xebia - Assessing the mental health wellbeing of Xebians.

Python version: 3.10
scikit-learn version: 1.7.2
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Load the Dataset
# ============================================================================
print("=" * 80)
print("STEP 1: Loading PHQ-9 Dataset")
print("=" * 80)

# Path to the dataset
dataset_path = os.path.join('..', 'data', 'PHQ-9_Dataset_5th Edition.csv')
if not os.path.exists(dataset_path):
    # Try current directory if not in ../data/
    dataset_path = 'PHQ-9_Dataset_5th Edition.csv'

try:
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded successfully from: {dataset_path}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Display basic info
print(f"\nDataset Shape: {df.shape}")
print("\nInitial Column Names:")
print(df.columns.tolist())

# ============================================================================
# STEP 2: Clean the Dataset
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Cleaning PHQ-9 Dataset")
print("=" * 80)

# 1. Trim extra spaces from column names
df.columns = df.columns.str.strip()
print("Column names trimmed.")

# 2. Standardize text values (case-insensitive) - strip and lowercase
# Also handle missing values safely
df_clean = df.copy()
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
        # Handle 'nan' strings that might result from previous step
        df_clean[col] = df_clean[col].replace('nan', np.nan)

# 3. Handle missing values safely
# Categorical -> Mode, Numeric -> Median
for col in df_clean.columns:
    if df_clean[col].isnull().sum() > 0:
        if df_clean[col].dtype == 'object':
            mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'unknown'
            df_clean[col] = df_clean[col].fillna(mode_val)
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# 4. Ensure PHQ_Severity and PHQ_Total are preserved
# (They are already in the dataframe, we just ensure they aren't dropped)
print(f"Preserved PHQ_Severity unique values: {df_clean['PHQ_Severity'].unique()}")

# ============================================================================
# STEP 3 & 4: Encode PHQ-9 Questionnaire Responses
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3 & 4: Encoding PHQ-9 Responses")
print("=" * 80)

phq_encoding = {
    "not at all": 0,
    "several days": 1,
    "more than half the days": 2,
    "nearly every day": 3
}

# Identify PHQ question columns (usually 9 items)
# Based on typical PHQ-9 datasets, these are often the middle columns
# We'll identify them by excluding metadata and targets
non_phq_cols = ['Age', 'Gender', 'PHQ_Total', 'PHQ_Severity', 'Sleep Quality', 'Study Pressure', 'Work Pressure', 'Financial Pressure']
phq_question_columns = [col for col in df_clean.columns if col not in non_phq_cols]

print(f"Detected {len(phq_question_columns)} PHQ question columns.")

# Apply encoding
for col in phq_question_columns:
    df_clean[col] = df_clean[col].map(phq_encoding)
    # Fill any unmapped values with 0 as a fallback
    df_clean[col] = df_clean[col].fillna(0).astype(int)

# Verify encoding
print("\nVerification of encoding (first 3 rows):")
print(df_clean[phq_question_columns].head(3))

# ============================================================================
# STEP 5 & 6: Encode Categorical Columns
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5 & 6: Encoding Categorical Columns")
print("=" * 80)

label_encoders = {}

# Gender: Label Encoding
if 'Gender' in df_clean.columns:
    le_gender = LabelEncoder()
    df_clean['Gender_encoded'] = le_gender.fit_transform(df_clean['Gender'])
    label_encoders['Gender'] = le_gender
    print("Gender encoded.")

# Sleep Quality: Ordinal Encoding
sleep_map = {'good': 0, 'average': 1, 'bad': 2, 'worst': 3}
if 'Sleep Quality' in df_clean.columns:
    df_clean['Sleep Quality_encoded'] = df_clean['Sleep Quality'].map(sleep_map).fillna(1).astype(int)
    print("Sleep Quality encoded (Ordinal).")

# Pressure columns: Handle both "Study Pressure" and "Work Pressure" if present
pressure_map = {'good': 0, 'average': 1, 'bad': 2, 'worst': 3}
for col in ['Study Pressure', 'Work Pressure', 'Financial Pressure']:
    if col in df_clean.columns:
        df_clean[f'{col}_encoded'] = df_clean[col].map(pressure_map).fillna(1).astype(int)
        print(f"{col} encoded (Ordinal).")

# ============================================================================
# STEP 7: Define Features and Target
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: Defining Features and Target")
print("=" * 80)

# Build feature list dynamically based on what was encoded
feature_cols = phq_question_columns + ['Age']
if 'Gender_encoded' in df_clean.columns: feature_cols.append('Gender_encoded')
if 'Sleep Quality_encoded' in df_clean.columns: feature_cols.append('Sleep Quality_encoded')
if 'Study Pressure_encoded' in df_clean.columns: feature_cols.append('Study Pressure_encoded')
if 'Work Pressure_encoded' in df_clean.columns: feature_cols.append('Work Pressure_encoded')
if 'Financial Pressure_encoded' in df_clean.columns: feature_cols.append('Financial Pressure_encoded')

X = df_clean[feature_cols]

# Target: PHQ_Severity
le_severity = LabelEncoder()
y = le_severity.fit_transform(df_clean['PHQ_Severity'])
label_encoders['PHQ_Severity'] = le_severity

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Severity mapping: {dict(zip(le_severity.classes_, le_severity.transform(le_severity.classes_)))}")

# ============================================================================
# STEP 8 & 12: Train PHQ-9 ML Models (Find Best)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8 & 12: Training and Selecting Best Model")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling for models like Logistic Regression and SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary to hold models and their performances
candidate_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

best_acc = 0
best_model = None
best_model_name = ""

for name, model in candidate_models.items():
    # Always train on scaled data for consistency and better convergence in most models
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_model_name = name

print(f"\nBest Model: {best_model_name} with Accuracy {best_acc:.4f}")

# Explainability
if best_model_name == "Logistic Regression":
    print("\nModel Coefficients (Logistic Regression):")
    coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': best_model.coef_[0]})
    print(coef_df.sort_values(by='Coefficient', ascending=False).head(10))
    print("\nNote: Logistic Regression was chosen for its interpretability and strong performance on categorical data.")
else:
    print("\nFeature Importance (Random Forest):")
    importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': best_model.feature_importances_})
    print(importance_df.sort_values(by='Importance', ascending=False).head(10))
    print("\nNote: Random Forest was chosen for its ability to handle non-linear relationships and provide feature importance.")

# ============================================================================
# STEP 9: Evaluate the Model
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: Evaluating the Best Model")
print("=" * 80)

# Always use scaled data for evaluation as all models are now trained on it
final_preds = best_model.predict(X_test_scaled)

print(f"Accuracy:  {accuracy_score(y_test, final_preds):.4f}")
print(f"Precision: {precision_score(y_test, final_preds, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, final_preds, average='weighted'):.4f}")
print(f"F1-score:  {f1_score(y_test, final_preds, average='weighted'):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, final_preds))

print("\nClassification Report:")
print(classification_report(y_test, final_preds, target_names=le_severity.classes_))

# ============================================================================
# STEP 10: Create Prediction Function (Microservice Style)
# ============================================================================

def predict_phq_severity(user_data):
    """
    Predict PHQ-9 severity for a single user.
    user_data example: {
        'responses': {'Item1': 'Several days', ...},
        'age': 25,
        'gender': 'Male',
        'sleep_quality': 'Average',
        'study_pressure': 'Bad',
        'financial_pressure': 'Good'
    }
    """
    # 1. Map responses
    encoded_responses = []
    # Note: Logic should match the feature_cols order exactly
    # For simplicity in this demo, we assume the keys in user_data['responses'] match
    # but in a real API, we'd ensure fixed order.
    
    # Use global/saved phq_question_columns
    for col in phq_question_columns:
        resp = user_data['responses'].get(col, 'not at all').lower().strip()
        encoded_responses.append(phq_encoding.get(resp, 0))
    
    # 2. Contextual features
    age = user_data.get('age', 25)
    gender = user_data.get('gender', 'male').lower().strip()
    sleep = user_data.get('sleep_quality', 'average').lower().strip()
    study = user_data.get('study_pressure', 'average').lower().strip()
    work = user_data.get('work_pressure', 'average').lower().strip()
    financial = user_data.get('financial_pressure', 'average').lower().strip()
    
    gender_enc = label_encoders['Gender'].transform([gender])[0] if 'Gender' in label_encoders else 0
    sleep_enc = sleep_map.get(sleep, 1)
    study_enc = pressure_map.get(study, 1)
    work_enc = pressure_map.get(work, 1)
    financial_enc = pressure_map.get(financial, 1)
    
    # 3. Assemble feature vector
    # Order: phq_questions, Age, Gender, Sleep, Study, Work, Financial
    features = encoded_responses + [age]
    if 'Gender_encoded' in feature_cols: features.append(gender_enc)
    if 'Sleep Quality_encoded' in feature_cols: features.append(sleep_enc)
    if 'Study Pressure_encoded' in feature_cols: features.append(study_enc)
    if 'Work Pressure_encoded' in feature_cols: features.append(work_enc)
    if 'Financial Pressure_encoded' in feature_cols: features.append(financial_enc)
    
    feature_arr = np.array(features).reshape(1, -1)
    
    # 4. Predict
    # Apply scaling consistently to all inputs before prediction
    feature_arr = scaler.transform(feature_arr)
    pred_idx = best_model.predict(feature_arr)[0]
    prob = np.max(best_model.predict_proba(feature_arr))
    
    severity_label = le_severity.inverse_transform([pred_idx])[0]
    
    return {
        'predicted_severity': severity_label,
        'confidence_score': float(prob)
    }

# ============================================================================
# STEP 11: Validate with Sample Input
# ============================================================================
print("\n" + "=" * 80)
print("STEP 11: Validating with Sample Input")
print("=" * 80)

sample_user = {
    'responses': {col: 'several days' for col in phq_question_columns},
    'age': 28,
    'gender': 'Female',
    'sleep_quality': 'Bad',
    'study_pressure': 'Average',
    'financial_pressure': 'Good'
}

result = predict_phq_severity(sample_user)
print(f"Sample Prediction: {result['predicted_severity']}")
print(f"Confidence: {result['confidence_score']:.4f}")

# ============================================================================
# Final: Save the Model and Scaler
# ============================================================================
model_package = {
    'model': best_model,
    'scaler': scaler,  # Always save the scaler for consistency
    'label_encoders': label_encoders,
    'severity_encoder': label_encoders.get('PHQ_Severity'), # Explicitly save severity encoder
    'feature_cols': feature_cols,
    'phq_question_columns': phq_question_columns,
    'phq_encoding': phq_encoding,
    'sleep_map': sleep_map,
    'pressure_map': pressure_map,
    'best_model_name': best_model_name
}

# Save the unified model package
joblib_path = os.path.join('..', 'models', 'phq9_best_model.joblib')
joblib.dump(model_package, joblib_path)
print(f"\nModel package saved to: {joblib_path}")

# Also save the scaler separately as requested
scaler_path = os.path.join('..', 'models', 'phq9_scaler.joblib')
joblib.dump(scaler, scaler_path)
print(f"Scaler saved separately to: {scaler_path}")
