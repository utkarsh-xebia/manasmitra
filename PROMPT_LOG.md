# Prompt Log

## Entry 1: PHQ-9 Mental Health Assessment ML Model

### Complete Prompt

You are a Machine Learning Engineer working with Xebia and you have to asses the mental health wellbeing of Xebians. Design a Machine learning model for assessing the mental health of xebians using PHQ 9 Questionnaire and the following dataset: 
Do the Following Steps:
 
1. Load the file "PHQ-9_Dataset_5th Edition.csv" present in the project using pandas.
 
Display:
 
- Column names
 
- First 5 rows
 
- Dataset shape
 
- Count of missing values per column
 

2. Clean the PHQ-9 dataset with the following rules:
 
- Trim extra spaces from column names
 
- Standardize text values (case-insensitive)
 
- Handle missing values safely (do not delete rows unless necessary)
 
- Ensure PHQ_Severity and PHQ_Total are preserved
 
3. Encode PHQ-9 questionnaire responses using standard scoring:
 
"Not at all" = 0   
 
"Several days" = 1   
 
"More than half the days" = 2   
 
"Nearly every day" = 3   
 

4. Apply this encoding only to the 9 PHQ question columns.
 
Verify encoding by printing sample rows.
 

5.Encode the following categorical columns into numeric values:
 
- Gender
 
- Sleep Quality
 
- Work Pressure
 
- Financial Pressure
 

6.Use label encoding or ordinal encoding where appropriate.
 
Explain in comments why each encoding choice was made.
7. Define Features and Target
 
Prepare the machine learning dataset:
 
 
- Define feature matrix X using encoded PHQ-9 questions and contextual features
 
- Define target variable y as PHQ_Severity
 
- Encode PHQ_Severity into numeric class labels
 
 
Print:
 
- Feature column names
 
- Target class mapping
 
8 — Train the PHQ-9 ML Model
 
Train an explainable classification model to predict PHQ_Severity:
 
- Split the data into training and testing sets
 
- Use Logistic Regression or Random Forest
 
- Train the model
 
- Print model coefficients or feature importance for explainability
 
Add comments explaining model choice.
 
9 — Evaluate the Model
 
Evaluate the trained PHQ-9 model:
 
- Print accuracy, precision, recall, and F1-score
 
- Display a confusion matrix
 
- Add brief comments interpreting the results
10 — Create Prediction Function (Microservice Style)
 
Create a reusable prediction function named `predict_phq_severity()`.
 
The function should:
 
- Accept a single user's PHQ-9 responses and contextual data
 
- Apply the same preprocessing and encoding steps
 
- Return:
 
   - Predicted PHQ severity label
 
   - Probability/confidence score
 
Ensure the function is modular and ready to be exposed as an API later.
 
11 — Validate with Sample Input
 
Test the PHQ-9 prediction function using a sample user input.
 
Print the predicted severity and confidence score clearly.

Update the PROMPT_LOG.md with the complete prompt (this one) → change made → impact/result.

### Changes Made

1. **Created `phq9_ml_model.py`**: A comprehensive Python script implementing all 11 steps of the PHQ-9 ML model pipeline.

2. **Dataset Loading (Step 1)**: 
   - Implemented pandas CSV loading with display of column names, first 5 rows, dataset shape, and missing value counts
   - Dataset has 682 rows and 16 columns with no missing values

3. **Data Cleaning (Step 2)**:
   - Trimmed extra spaces from column names
   - Standardized text values to lowercase and trimmed whitespace
   - Implemented safe missing value handling (using mode for categorical, median for numeric)
   - Preserved PHQ_Severity and PHQ_Total columns

4. **PHQ-9 Response Encoding (Steps 3-4)**:
   - Implemented encoding mapping: "Not at all"=0, "Several days"=1, "More than half the days"=2, "Nearly every day"=3
   - Dynamically identified 9 PHQ question columns by excluding known non-PHQ columns
   - Applied encoding to all PHQ question columns with verification

5. **Categorical Encoding (Steps 5-6)**:
   - **Gender**: Used Label Encoding (nominal categorical, no inherent order)
   - **Sleep Quality**: Used Ordinal Encoding (inherent order: Good < Average < Bad < Worst)
   - **Study Pressure**: Used Ordinal Encoding (inherent order: Good < Average < Bad < Worst)
   - **Financial Pressure**: Used Ordinal Encoding (inherent order: Good < Average < Bad < Worst)
   - Note: Dataset contains "Study Pressure" not "Work Pressure" as mentioned in prompt

6. **Feature and Target Definition (Step 7)**:
   - Feature matrix X includes: 9 encoded PHQ questions + Age + encoded categorical variables (14 features total)
   - Target variable y is PHQ_Severity encoded to numeric labels (5 classes: mild, minimal, moderate, moderately severe, severe)
   - Printed feature column names and target class mapping

7. **Model Training (Step 8)**:
   - Selected Random Forest Classifier for explainability (feature importance) and robust performance
   - Split data: 80% training (545 samples), 20% testing (137 samples) with stratification
   - Trained model with 100 estimators, max_depth=10, random_state=42
   - Displayed feature importance for model interpretability

8. **Model Evaluation (Step 9)**:
   - Achieved metrics: Accuracy: 83.21%, Precision: 83.77%, Recall: 83.21%, F1-Score: 82.62%
   - Generated confusion matrix and detailed classification report per severity class
   - Added interpretation comments explaining model performance

9. **Prediction Function (Step 10)**:
   - Created `predict_phq_severity()` function in microservice style
   - Function accepts: user_responses (dict), age, gender, sleep_quality, study_pressure, financial_pressure
   - Returns: predicted_severity (label), confidence_score (float), severity_probabilities (dict)
   - Function is modular and ready for API integration
   - Applies same preprocessing and encoding as training data

10. **Validation (Step 11)**:
    - Tested prediction function with sample input
    - Sample prediction: "minimal" severity with 99.00% confidence
    - Verified function works correctly with proper output formatting

### Impact/Result

- **Model Performance**: The Random Forest model achieved 83.21% accuracy, demonstrating strong predictive capability for PHQ-9 severity classification
- **Explainability**: Feature importance shows that PHQ-9 questions (especially "Little interest or pleasure in doing things" and "Poor appetite or overeating") are the most important predictors, followed by contextual factors like Study Pressure and Sleep Quality
- **Production Ready**: The prediction function is designed in a microservice style, making it easy to integrate into an API endpoint for real-time mental health assessments
- **Code Quality**: Comprehensive implementation with clear comments, proper error handling, and structured output following all 11 steps
- **Dataset Insight**: Successfully processed 682 records with 5 severity classes, showing balanced performance across different severity levels (best performance on "minimal" and "moderately severe" classes)

The implementation provides a complete end-to-end solution for assessing mental health wellbeing using PHQ-9 questionnaire data, ready for deployment and further integration.
