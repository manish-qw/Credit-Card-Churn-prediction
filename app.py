from flask import Flask, request, jsonify
from flask import render_template
import joblib
import pandas as pd
import numpy as np

# Load model and encoders
model = joblib.load('xgb_model.pkl')
label_encoders = joblib.load('label_encoder.pkl')

app = Flask(__name__)


# Preprocessing Function (replicates your training logic)
def preprocess_input(data):
    # Create a DataFrame
    df = pd.DataFrame([data])

    # Feature Engineering
    if 'Total_Revolving_Bal' in df.columns and 'Credit_Limit' in df.columns:
        df['Utilization_Rate'] = df['Total_Revolving_Bal'] / df['Credit_Limit']
    if 'Months_on_book' in df.columns:
        df['Tenure_Group'] = pd.cut(df['Months_on_book'], bins=[0, 24, 36, 48, 60], labels=['0-24','25-36','37-48','49-60'])
    if 'Total_Trans_Ct' in df.columns and 'Total_Relationship_Count' in df.columns:
        df['Transaction_Relationship_Interaction'] = df['Total_Trans_Ct'] * df['Total_Relationship_Count']

    # Label Encoding for categorical features
    for col, encoder in label_encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col].astype(str))

    return df

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {key: request.form[key] for key in request.form}

        # List all numeric fields
        numeric_fields = [
            'Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
            'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct',
            'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio', 'Utilization_Rate',
            'Transaction_Relationship_Interaction'
        ]

        for num_field in numeric_fields:
            input_data[num_field] = float(input_data[num_field])

        processed = preprocess_input(input_data)
        # Ensure columns are in the same order as model expects
        expected_cols = [
            'Customer_Age', 'Gender', 'Dependent_count', 'Education_Level', 'Marital_Status',
            'Income_Category', 'Card_Category', 'Months_on_book', 'Total_Relationship_Count',
            'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct',
            'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio', 'Utilization_Rate',
            'Tenure_Group', 'Transaction_Relationship_Interaction'
        ]
        processed = processed.reindex(columns=expected_cols, fill_value=0)

        prediction = model.predict(processed)[0]
        result = "Customer will Churn" if prediction == 1 else "Customer will Stay"
        return render_template('index.html', prediction=result)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)