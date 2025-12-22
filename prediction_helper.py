import pandas as pd
from joblib import load
from xgboost import XGBModel

# --- Load models ---
model_rest = load("Artifacts/model_rest.joblib")
model_young = load("Artifacts/model_young.joblib")

if not hasattr(model_rest, 'predictor'):
    model_rest.predictor = None

if not hasattr(model_rest, 'gpu_id'):
    model_rest.gpu_id = None  # Or use 0 if you are actually using a GPU

# --- Load scalers (with columns) ---
scaler_rest_dict = load("Artifacts/scaler_rest.joblib")
scaler_rest = scaler_rest_dict['scaler']
cols_rest = scaler_rest_dict['cols_to_scale']

scaler_young_dict = load("Artifacts/scaler_young.joblib")
scaler_young = scaler_young_dict['scaler']
cols_young = scaler_young_dict['cols_to_scale']


# --- Preprocessing function ---
def preprocess_input(input_data, scaler_young, cols_young, scaler_rest, cols_rest):
    # Rename numeric columns
    input_data = input_data.rename(columns={
        'Age': 'age',
        'Number Of Dependants': 'number_of_dependants',
        'Income_Lakhs': 'income_lakhs',
        'Insurance_Plan': 'insurance_plan'
    })

    # Map income_level
    input_data['insurance_plan'] = input_data['insurance_plan'].map({'Bronze' : 1,
                                                   'Silver' : 2,
                                                   'Gold' : 3
                                                  })

    # Split medical history
    def split_diseases(x):
        diseases = x.lower().split(' & ')
        if len(diseases) == 1:
            return pd.Series([diseases[0], 'none'])
        else:
            return pd.Series(diseases)

    input_data[['disease1', 'disease2']] = input_data['Medical History'].apply(split_diseases)

    # Risk scores
    risk_scores = {'diabetes': 6, 'high blood pressure': 6, 'heart disease': 8, 'thyroid': 5, 'no disease': 0,
                   'none': 0}
    input_data['total_risk_score'] = input_data['disease1'].map(risk_scores) + input_data['disease2'].map(risk_scores)

    # Normalized risk score
    input_data['normalized_risk_score'] = input_data['total_risk_score'] / 14

    # One-hot encode nominal columns
    nominal_cols = {'Gender': 'gender', 'Region': 'region', 'Marital_status': 'marital_status',
                    'BMI_Category': 'bmi_category',
                    'Smoking_Status': 'smoking_status', 'Employment_Status': 'employment_status'}
    input_data = input_data.rename(columns=nominal_cols)
    input_data = pd.get_dummies(input_data, columns=nominal_cols.values(), drop_first=True, dtype=int)

    # Drop unnecessary columns
    input_data = input_data.drop(['Medical History', 'disease1', 'disease2', 'total_risk_score'], axis=1)

    # Conditional scaling
    input_data['income_level'] = None
    if input_data['age'].iloc[0] <= 25:
        input_data[cols_young] = scaler_young.transform(input_data[cols_young])
    else:
        input_data[cols_rest] = scaler_rest.transform(input_data[cols_rest])
    input_data.drop('income_level', axis=1, inplace=True)
    # Align to expected columns
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight',
        'smoking_status_Occasional', 'smoking_status_Regular', 'employment_status_Salaried',
        'employment_status_Self-Employed'
    ]
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    return input_data[expected_columns]


# --- Predict function ---
def predict(input_data):
    # Choose model based on age
    if input_data['Age'].iloc[0] <= 25:
        model = model_young
        df = preprocess_input(input_data, scaler_young, cols_young, scaler_rest, cols_rest)
    else:
        model = model_rest
        df = preprocess_input(input_data, scaler_young, cols_young, scaler_rest, cols_rest)

    prediction = model.predict(df)
    return prediction[0]