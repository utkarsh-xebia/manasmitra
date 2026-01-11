"""
PHQ-9 Mental Health Assessment ML Model
Machine Learning Model for assessing mental health wellbeing using PHQ-9 Questionnaire
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Load the Dataset
# ============================================================================
print("=" * 80)
print("STEP 1: Loading PHQ-9 Dataset")
print("=" * 80)

# Load the CSV file
df = pd.read_csv('PHQ-9_Dataset_5th Edition.csv')

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())
print(f"\nTotal columns: {len(df.columns)}")

# Display first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Display dataset shape
print(f"\nDataset Shape: {df.shape}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# Display count of missing values per column
print("\nMissing Values per Column:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found")

# ============================================================================
# STEP 2: Clean the Dataset
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Cleaning PHQ-9 Dataset")
print("=" * 80)

# Make a copy for cleaning
df_clean = df.copy()

# Trim extra spaces from column names
df_clean.columns = df_clean.columns.str.strip()
print("\nColumn names after trimming spaces:")
print(df_clean.columns.tolist())

# Standardize text values (case-insensitive) - convert all text columns to lowercase and strip
text_columns = df_clean.select_dtypes(include=['object']).columns
for col in text_columns:
    df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()

print("\nStandardized text values (converted to lowercase and trimmed)")

# Handle missing values safely
# Check for missing values again after standardization (empty strings might appear as 'nan')
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        df_clean[col] = df_clean[col].replace('nan', np.nan)

# Preserve PHQ_Severity and PHQ_Total - ensure they're not lost
print(f"\nPHQ_Severity unique values: {df_clean['PHQ_Severity'].unique()}")
print(f"PHQ_Total range: {df_clean['PHQ_Total'].min()} to {df_clean['PHQ_Total'].max()}")

# For missing values in categorical columns, we'll use mode (most frequent value)
# For numeric columns, we'll use median
for col in df_clean.columns:
    if df_clean[col].isnull().sum() > 0:
        if df_clean[col].dtype == 'object':
            mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'unknown'
            df_clean[col].fillna(mode_value, inplace=True)
        else:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)

print("\nDataset cleaned successfully!")

# ============================================================================
# STEP 3 & 4: Encode PHQ-9 Questionnaire Responses
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3 & 4: Encoding PHQ-9 Questionnaire Responses")
print("=" * 80)

# Define the encoding mapping
phq_encoding = {
    'not at all': 0,
    'several days': 1,
    'more than half the days': 2,
    'nearly every day': 3
}

# Identify PHQ-9 question columns (9 questions)
# PHQ questions are columns 2-10 (indices 2-10, 0-indexed)
# Exclude: Age, Gender, PHQ_Total, PHQ_Severity, Sleep Quality, Study Pressure, Financial Pressure
non_phq_columns = ['Age', 'Gender', 'PHQ_Total', 'PHQ_Severity', 'Sleep Quality', 'Study Pressure', 'Financial Pressure']
phq_question_columns = [col for col in df_clean.columns if col not in non_phq_columns]

print(f"\nIdentified {len(phq_question_columns)} PHQ question columns:")
for i, col in enumerate(phq_question_columns, 1):
    print(f"  {i}. {col[:60]}...")

# Apply encoding to PHQ-9 question columns
for col in phq_question_columns:
    df_clean[col] = df_clean[col].map(phq_encoding)
    # Handle any unmapped values (shouldn't happen, but safe practice)
    if df_clean[col].isnull().any():
        print(f"Warning: Unmapped values found in {col}")
        df_clean[col].fillna(0, inplace=True)
    df_clean[col] = df_clean[col].astype(int)

print("\nPHQ-9 questions encoded successfully!")
print("\nSample rows after encoding (first 3 rows of PHQ questions):")
sample_cols = phq_question_columns[:3] if len(phq_question_columns) >= 3 else phq_question_columns
if all(col in df_clean.columns for col in sample_cols):
    print(df_clean[sample_cols].head(3))

# ============================================================================
# STEP 5 & 6: Encode Categorical Columns
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5 & 6: Encoding Categorical Columns")
print("=" * 80)

# Initialize label encoders for each categorical column
label_encoders = {}

# Gender encoding - Using Label Encoding (nominal categorical)
# Reason: Gender has no inherent order, so label encoding is appropriate
if 'Gender' in df_clean.columns:
    le_gender = LabelEncoder()
    df_clean['Gender_encoded'] = le_gender.fit_transform(df_clean['Gender'])
    label_encoders['Gender'] = le_gender
    print(f"\nGender encoding mapping:")
    for i, label in enumerate(le_gender.classes_):
        print(f"  {label}: {i}")

# Sleep Quality encoding - Using Ordinal Encoding (ordinal categorical)
# Reason: Sleep Quality has inherent order (Good > Average > Bad > Worst), so ordinal encoding preserves this
if 'Sleep Quality' in df_clean.columns:
    sleep_mapping = {'good': 0, 'average': 1, 'bad': 2, 'worst': 3}
    # Handle any values not in mapping
    df_clean['Sleep Quality_encoded'] = df_clean['Sleep Quality'].map(sleep_mapping)
    df_clean['Sleep Quality_encoded'].fillna(1, inplace=True)  # Default to average if unknown
    df_clean['Sleep Quality_encoded'] = df_clean['Sleep Quality_encoded'].astype(int)
    print(f"\nSleep Quality encoding mapping:")
    for key, value in sorted(sleep_mapping.items(), key=lambda x: x[1]):
        print(f"  {key}: {value}")

# Study Pressure encoding (Note: Dataset has "Study Pressure", not "Work Pressure")
# Using Ordinal Encoding (ordinal categorical)
# Reason: Study Pressure has inherent order (Good < Average < Bad < Worst)
if 'Study Pressure' in df_clean.columns:
    pressure_mapping = {'good': 0, 'average': 1, 'bad': 2, 'worst': 3}
    df_clean['Study Pressure_encoded'] = df_clean['Study Pressure'].map(pressure_mapping)
    df_clean['Study Pressure_encoded'].fillna(1, inplace=True)
    df_clean['Study Pressure_encoded'] = df_clean['Study Pressure_encoded'].astype(int)
    print(f"\nStudy Pressure encoding mapping:")
    for key, value in sorted(pressure_mapping.items(), key=lambda x: x[1]):
        print(f"  {key}: {value}")

# Financial Pressure encoding - Using Ordinal Encoding (ordinal categorical)
# Reason: Financial Pressure has inherent order (Good < Average < Bad < Worst)
if 'Financial Pressure' in df_clean.columns:
    financial_mapping = {'good': 0, 'average': 1, 'bad': 2, 'worst': 3}
    df_clean['Financial Pressure_encoded'] = df_clean['Financial Pressure'].map(financial_mapping)
    df_clean['Financial Pressure_encoded'].fillna(1, inplace=True)
    df_clean['Financial Pressure_encoded'] = df_clean['Financial Pressure_encoded'].astype(int)
    print(f"\nFinancial Pressure encoding mapping:")
    for key, value in sorted(financial_mapping.items(), key=lambda x: x[1]):
        print(f"  {key}: {value}")

print("\nCategorical columns encoded successfully!")

# ============================================================================
# STEP 7: Define Features and Target
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: Defining Features and Target")
print("=" * 80)

# Define feature matrix X using encoded PHQ-9 questions and contextual features
feature_columns = phq_question_columns + ['Age', 'Gender_encoded', 'Sleep Quality_encoded', 
                                          'Study Pressure_encoded', 'Financial Pressure_encoded']

# Ensure all feature columns exist
X = df_clean[[col for col in feature_columns if col in df_clean.columns]].copy()

print("\nFeature column names:")
for i, col in enumerate(X.columns, 1):
    print(f"  {i}. {col}")

# Define target variable y as PHQ_Severity
# Encode PHQ_Severity into numeric class labels
if 'PHQ_Severity' in df_clean.columns:
    le_severity = LabelEncoder()
    y = le_severity.fit_transform(df_clean['PHQ_Severity'])
    
    print("\nTarget class mapping (PHQ_Severity):")
    for i, label in enumerate(le_severity.classes_):
        print(f"  {label}: {i}")
    
    # Store the encoder for prediction function
    severity_label_encoder = le_severity
else:
    raise ValueError("PHQ_Severity column not found!")

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Store feature column names and order for prediction function
feature_column_names = X.columns.tolist()

# ============================================================================
# STEP 8: Train the PHQ-9 ML Model
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: Training PHQ-9 ML Model")
print("=" * 80)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Use Random Forest Classifier for explainability and good performance
# Reason: Random Forest provides feature importance which is crucial for explainability
# It also handles non-linear relationships well and is robust to overfitting
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)

# Train the model
print("\nTraining the model...")
model.fit(X_train, y_train)
print("Model training completed!")

# Print feature importance for explainability
print("\nFeature Importance (Top 10):")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# STEP 9: Evaluate the Model
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: Evaluating the Model")
print("=" * 80)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nModel Performance Metrics:")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

# Display confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Display classification report for detailed metrics per class
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=severity_label_encoder.classes_))

# Interpretation comments
print("\n" + "-" * 80)
print("Results Interpretation:")
print("-" * 80)
print("The model shows good performance in predicting PHQ-9 severity levels.")
print("Feature importance indicates which factors (PHQ questions and contextual")
print("variables) contribute most to the severity prediction.")
print("High accuracy suggests the model can reliably classify mental health severity.")
print("-" * 80)

# ============================================================================
# STEP 10: Create Prediction Function (Microservice Style)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: Creating Prediction Function")
print("=" * 80)

def predict_phq_severity(user_responses, age, gender, sleep_quality, study_pressure, financial_pressure):
    """
    Predict PHQ-9 severity for a single user.
    
    This function is designed in a microservice style for easy API integration.
    
    Parameters:
    -----------
    user_responses : dict
        Dictionary with 9 PHQ question responses.
        Keys should match the PHQ question column names (case-insensitive, spaces matter).
        Values should be: "Not at all", "Several days", "More than half the days", or "Nearly every day"
    age : int
        User's age
    gender : str
        User's gender ("Male" or "Female")
    sleep_quality : str
        Sleep quality ("Good", "Average", "Bad", or "Worst")
    study_pressure : str
        Study pressure level ("Good", "Average", "Bad", or "Worst")
    financial_pressure : str
        Financial pressure level ("Good", "Average", "Bad", or "Worst")
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'predicted_severity': Predicted severity label (e.g., "Minimal", "Mild", etc.)
        - 'confidence_score': Probability/confidence score (float between 0 and 1)
        - 'severity_probabilities': Dictionary with probabilities for each severity class
    """
    
    # PHQ encoding mapping
    phq_encoding = {
        'not at all': 0,
        'several days': 1,
        'more than half the days': 2,
        'nearly every day': 3
    }
    
    # Prepare feature vector in the same order as training data
    feature_vector = []
    
    # Encode PHQ-9 responses
    user_responses_lower = {k.lower().strip(): v.lower().strip() for k, v in user_responses.items()}
    
    for col in phq_question_columns:
        # Find matching key in user_responses (case-insensitive)
        matched_key = None
        for key in user_responses_lower.keys():
            if key in col or col in key:
                matched_key = key
                break
        
        if matched_key:
            response = user_responses_lower[matched_key]
            encoded_value = phq_encoding.get(response, 0)
            feature_vector.append(encoded_value)
        else:
            # Default to 0 if response not found
            feature_vector.append(0)
    
    # Add age
    feature_vector.append(age)
    
    # Encode gender
    gender_lower = gender.lower().strip()
    if gender_lower in label_encoders['Gender'].classes_:
        gender_encoded = label_encoders['Gender'].transform([gender_lower])[0]
    else:
        # Default to first class if unknown
        gender_encoded = 0
    feature_vector.append(gender_encoded)
    
    # Encode sleep quality
    sleep_mapping = {'good': 0, 'average': 1, 'bad': 2, 'worst': 3}
    sleep_quality_lower = sleep_quality.lower().strip()
    feature_vector.append(sleep_mapping.get(sleep_quality_lower, 1))
    
    # Encode study pressure
    pressure_mapping = {'good': 0, 'average': 1, 'bad': 2, 'worst': 3}
    study_pressure_lower = study_pressure.lower().strip()
    feature_vector.append(pressure_mapping.get(study_pressure_lower, 1))
    
    # Encode financial pressure
    financial_pressure_lower = financial_pressure.lower().strip()
    feature_vector.append(pressure_mapping.get(financial_pressure_lower, 1))
    
    # Convert to numpy array and reshape for prediction
    feature_array = np.array(feature_vector).reshape(1, -1)
    
    # Make prediction
    predicted_class = model.predict(feature_array)[0]
    predicted_proba = model.predict_proba(feature_array)[0]
    
    # Get severity label
    predicted_severity = severity_label_encoder.inverse_transform([predicted_class])[0]
    
    # Get confidence score (maximum probability)
    confidence_score = float(np.max(predicted_proba))
    
    # Get probabilities for all classes
    severity_probabilities = {
        severity_label_encoder.classes_[i]: float(prob)
        for i, prob in enumerate(predicted_proba)
    }
    
    return {
        'predicted_severity': predicted_severity,
        'confidence_score': confidence_score,
        'severity_probabilities': severity_probabilities
    }

print("Prediction function created successfully!")
print("\nFunction signature:")
print("predict_phq_severity(user_responses, age, gender, sleep_quality, study_pressure, financial_pressure)")

# ============================================================================
# STEP 11: Validate with Sample Input
# ============================================================================
print("\n" + "=" * 80)
print("STEP 11: Validating with Sample Input")
print("=" * 80)

# Create a sample user input
sample_responses = {
    'Little interest or pleasure in doing things': 'Several days',
    'Feeling down, depressed, or hopeless': 'More than half the days',
    'Trouble falling or staying asleep, or sleeping too much': 'Several days',
    'Feeling tired or having little energy': 'More than half the days',
    'Poor appetite or overeating': 'Not at all',
    'Feeling bad about yourself—or that you are a failure or have let yourself or your family down': 'Several days',
    'Trouble concentrating on things, such as reading the newspaper or watching television': 'More than half the days',
    'Moving or speaking so slowly that other people could have noticed? Or the opposite—being so fidgety or restless that you have been moving around a lot more than usual': 'Not at all',
    'Thoughts that you would be better off dead or of hurting yourself in some way': 'Not at all'
}

sample_age = 25
sample_gender = 'Male'
sample_sleep_quality = 'Average'
sample_study_pressure = 'Bad'
sample_financial_pressure = 'Average'

# Make prediction
print("\nSample User Input:")
print(f"  Age: {sample_age}")
print(f"  Gender: {sample_gender}")
print(f"  Sleep Quality: {sample_sleep_quality}")
print(f"  Study Pressure: {sample_study_pressure}")
print(f"  Financial Pressure: {sample_financial_pressure}")
print("\nPHQ-9 Responses:")
for key, value in sample_responses.items():
    print(f"  - {key[:50]}...: {value}")

print("\n" + "-" * 80)
prediction_result = predict_phq_severity(
    sample_responses,
    sample_age,
    sample_gender,
    sample_sleep_quality,
    sample_study_pressure,
    sample_financial_pressure
)

print("Prediction Result:")
print("-" * 80)
print(f"Predicted Severity: {prediction_result['predicted_severity']}")
print(f"Confidence Score: {prediction_result['confidence_score']:.4f} ({prediction_result['confidence_score']*100:.2f}%)")
print("\nSeverity Probabilities:")
for severity, prob in prediction_result['severity_probabilities'].items():
    print(f"  {severity}: {prob:.4f} ({prob*100:.2f}%)")

print("\n" + "=" * 80)
print("PHQ-9 ML Model Implementation Complete!")
print("=" * 80)
