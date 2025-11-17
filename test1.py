# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

uploaded = st.file_uploader("C:\\Users\\user\\Downloads\\Assignment\\Assignment\\salary_data.csv", type=["csv"])
# --- 1. Load CSV ---
df = pd.read_csv(uploaded)

# --- 2. Features & Target ---
X = df[['work_year', 'experience_level', 'employment_type', 'job_title',
        'employee_residence', 'remote_ratio', 'company_location', 'company_size']]
y = df['salary_in_usd']

# --- 3. Preprocessing ---
categorical_features = ['experience_level', 'employment_type', 'job_title',
                        'employee_residence', 'company_location', 'company_size']
numeric_features = ['work_year', 'remote_ratio']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough'
)

# --- 4. Model ---
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X, y)

# --- 5. Streamlit App ---
st.title("Salary Prediction Web App")

st.sidebar.header("Input Features")

def user_input():
    data = {
        'work_year': st.sidebar.selectbox('Work Year', [2021, 2022, 2023, 2024, 2025]),
        'experience_level': st.sidebar.selectbox('Experience Level', df['experience_level'].unique()),
        'employment_type': st.sidebar.selectbox('Employment Type', df['employment_type'].unique()),
        'job_title': st.sidebar.selectbox('Job Title', df['job_title'].unique()),
        'employee_residence': st.sidebar.selectbox('Employee Residence', df['employee_residence'].unique()),
        'remote_ratio': st.sidebar.slider('Remote Ratio (%)', 0, 100, 50),
        'company_location': st.sidebar.selectbox('Company Location', df['company_location'].unique()),
        'company_size': st.sidebar.selectbox('Company Size', df['company_size'].unique())
    }
    return pd.DataFrame([data])

input_df = user_input()

# --- 6. Prediction ---
prediction = model.predict(input_df)

st.subheader("Predicted Salary (USD)")
st.write(f"${prediction[0]:,.2f}")

# --- 7. Optional: Trend Plot ---
import matplotlib.pyplot as plt

st.subheader("Salary Trend (Random Sample Jobs)")

sample_jobs = df['job_title'].unique()[:3]  # pick 3 job titles
for job in sample_jobs:
    job_df = df[df['job_title'] == job]
    years = list(range(2021, 2026))
    salaries = []
    for year in years:
        input_sample = job_df.iloc[0].copy()
        input_sample['work_year'] = year
        salaries.append(model.predict(pd.DataFrame([input_sample]))[0])
    plt.plot(years, salaries, label=job)

plt.xlabel('Year')
plt.ylabel('Predicted Salary (USD)')
plt.legend()
st.pyplot(plt)





