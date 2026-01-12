import sys
import json
import joblib
import numpy as np
import os

def predict():
    try:
        # Load the model package
        # Assuming the script is called from backend directory or absolute path is provided
        model_path = os.path.join(os.getcwd(), '..', 'models', 'phq9_best_model.joblib')
        if not os.path.exists(model_path):
            # Try another common path
            model_path = os.path.join(os.getcwd(), 'models', 'phq9_best_model.joblib')
        
        model_package = joblib.load(model_path)
        
        model = model_package['model']
        scaler = model_package['scaler']
        label_encoders = model_package['label_encoders']
        severity_encoder = model_package.get('severity_encoder', label_encoders.get('PHQ_Severity'))
        phq_question_columns = model_package['phq_question_columns']
        
        # Check if severity_encoder exists
        if severity_encoder is None:
            raise KeyError("Severity encoder not found in model package")

        # Read input from stdin
        input_data = json.load(sys.stdin)
        
        answers = input_data.get('answers', [])
        age = input_data.get('age', 25)
        gender = input_data.get('gender', 'male').lower()
        sleep_quality = input_data.get('sleepQuality', 'average').lower()
        study_pressure = input_data.get('studyPressure', 'average').lower()
        financial_pressure = input_data.get('financialPressure', 'average').lower()

        # Prepare feature vector
        # Order: 9 PHQ questions, Age, Gender_encoded, Sleep Quality_encoded, Study Pressure_encoded, Financial Pressure_encoded
        feature_vector = []
        feature_vector.extend(answers)
        feature_vector.append(age)
        
        # Encode gender
        gender_encoded = 0
        if 'Gender' in label_encoders:
            try:
                gender_encoded = label_encoders['Gender'].transform([gender])[0]
            except:
                gender_encoded = 0
        feature_vector.append(gender_encoded)
        
        # Mappings used in training
        sleep_mapping = {'good': 0, 'average': 1, 'bad': 2, 'worst': 3}
        pressure_mapping = {'good': 0, 'average': 1, 'bad': 2, 'worst': 3}
        
        feature_vector.append(sleep_mapping.get(sleep_quality, 1))
        feature_vector.append(pressure_mapping.get(study_pressure, 1))
        feature_vector.append(pressure_mapping.get(financial_pressure, 1))

        # Convert to numpy array
        feature_array = np.array(feature_vector).reshape(1, -1)
        
        # Scale if needed
        if scaler:
            feature_array = scaler.transform(feature_array)
            
        # Predict
        predicted_class = model.predict(feature_array)[0]
        probabilities = model.predict_proba(feature_array)[0]
        
        severity_label = severity_encoder.inverse_transform([predicted_class])[0]
        confidence_score = float(np.max(probabilities))
        
        # Map severity labels to title case for frontend
        severity_label = severity_label.capitalize()
        if severity_label == 'Moderately severe':
            severity_label = 'Moderately Severe'

        # Calculate derived metrics
        total_score = sum(answers)
        
        # Stress level: Based on study/financial pressure and PHQ score (0-10 scale)
        # Higher score = more stress
        pressure_val = (pressure_mapping.get(study_pressure, 1) + pressure_mapping.get(financial_pressure, 1)) / 2
        stress_level = round(min(10, (total_score / 27 * 6) + (pressure_val / 3 * 4)), 1)
        
        # Mood score: Inverse of PHQ score (0-10 scale)
        mood_score = round(max(0, 10 - (total_score / 27 * 10)), 1)
        
        # Work-life balance: Based on sleep quality and pressure (0-10 scale)
        sleep_val = sleep_mapping.get(sleep_quality, 1)
        work_life_balance = round(max(0, 10 - (sleep_val / 3 * 5) - (pressure_val / 3 * 5)), 1)

        # Generate recommendations
        recommendations = []
        if total_score >= 15:
            recommendations = [
                "Schedule a meeting with a mental health professional",
                "Consider taking a few days of 'Wellness Leave'",
                "Practice intensive mindfulness meditation (20+ mins)",
                "Reach out to HR about available Employee Assistance Programs (EAP)"
            ]
        elif total_score >= 10:
            recommendations = [
                "Practice daily mindfulness for 10 minutes",
                "Set strict boundaries for work-life balance",
                "Engage in at least 30 minutes of physical exercise",
                "Talk to a trusted colleague or mentor about your workload"
            ]
        else:
            recommendations = [
                "Maintain your current healthy habits",
                "Try a new hobby or creative activity this week",
                "Get 7-8 hours of consistent sleep",
                "Practice daily gratitude journaling"
            ]

        # Return results
        result = {
            'severityLabel': severity_label,
            'confidenceScore': confidence_score,
            'totalScore': total_score,
            'stressLevel': stress_level,
            'moodScore': mood_score,
            'workLifeBalance': work_life_balance,
            'recommendations': recommendations
        }
        
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({'error': str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    predict()
