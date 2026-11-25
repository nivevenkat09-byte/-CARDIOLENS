import streamlit as st
import pandas as pd
import sqlite3, joblib, os
from datetime import datetime

st.set_page_config(page_title='CardioLens', page_icon='❤️', layout='wide')

FILES = ['heart_disease_model.pkl', 'heart_disease_demo_data.csv', 'heart_disease_project.db']
missing = [f for f in FILES[:2] if not os.path.exists(f)]
if missing:
    st.error('Missing required project file(s): ' + ', '.join(missing))
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load('heart_disease_model.pkl')

model = load_model()
conn = sqlite3.connect('heart_disease_project.db', check_same_thread=False)
cur = conn.cursor()
cur.execute('''CREATE TABLE IF NOT EXISTS predictions (
 id INTEGER PRIMARY KEY AUTOINCREMENT, age INTEGER, sex INTEGER, bmi REAL,
 blood_pressure INTEGER, diabetes INTEGER, smoker INTEGER, general_health INTEGER,
 predicted_class INTEGER, created_at TEXT)''')
cur.execute('''CREATE TABLE IF NOT EXISTS model_results (
 id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT, accuracy REAL, precision REAL,
 recall REAL, f1_score REAL, created_at TEXT)''')
conn.commit()

st.title('❤️ CardioLens')
st.subheader('Exploring Cardiovascular Health Patterns Through Machine Learning')
st.caption('Interactive classification demonstration using CDC BRFSS 2023 survey data')
st.warning('Educational demonstration only: this classifies survey-reported history and is not a diagnosis or prediction of future cardiac events.')

st.header('1. Model Performance')
results = pd.read_sql_query('SELECT model_name, accuracy, precision, recall, f1_score FROM model_results', conn)
if results.empty:
    st.info('Model comparison values are not stored in this database. The trained model is still available for predictions.')
else:
    st.dataframe(results, use_container_width=True)

st.header('2. Enter Survey Details')
with st.form('cardiolens_form'):
    left, right = st.columns(2)
    with left:
        age = st.slider('Age (top-coded at 80)', 18, 80, 45)
        sex_label = st.selectbox('Sex', ['Male', 'Female'])
        bmi = st.number_input('BMI', min_value=10.0, max_value=80.0, value=25.0, step=0.1)
        bp_label = st.selectbox('High blood pressure status', ['Yes', 'No', 'Not sure'])
    with right:
        diabetes_label = st.selectbox('Diabetes status', ['Yes', 'No', 'Prediabetes', 'Only during pregnancy'])
        smoker_label = st.selectbox('Smoking status', ['Current daily smoker', 'Current some-days smoker', 'Former smoker', 'Never smoked'])
        health_label = st.selectbox('General health', ['Excellent', 'Very good', 'Good', 'Fair', 'Poor'])
    submitted = st.form_submit_button('Run Prediction', type='primary')

# Codes follow the commonly used BRFSS response categories; verify against the specific derived-variable codebook.
sex_code = 1 if sex_label == 'Male' else 2
bp_code = {'Yes': 1, 'No': 2, 'Not sure': 7}[bp_label]
diabetes_code = {'Yes': 1, 'Only during pregnancy': 2, 'No': 3, 'Prediabetes': 4}[diabetes_label]
smoker_code = {'Current daily smoker': 1, 'Current some-days smoker': 2, 'Former smoker': 3, 'Never smoked': 4}[smoker_label]
health_code = {'Excellent': 1, 'Very good': 2, 'Good': 3, 'Fair': 4, 'Poor': 5}[health_label]

if submitted:
    input_df = pd.DataFrame([{
        '_AGE80': float(age), '_SEX': float(sex_code), '_BMI5': float(round(bmi * 100)),
        'BPHIGH6': float(bp_code), 'DIABETE4': float(diabetes_code),
        '_SMOKER3': float(smoker_code), 'GENHLTH': float(health_code)
    }])
    try:
        pred = int(model.predict(input_df)[0])
        score = float(model.predict_proba(input_df)[0][1])
        st.header('3. Prediction Result')
        if pred == 1:
            st.error('Model classification: Reported history of heart attack or coronary heart disease')
        else:
            st.success('Model classification: No reported history of heart attack or coronary heart disease')
        st.metric('Model score for class 1', f'{score:.1%}')
        st.caption('This score is not a clinical risk estimate or a calibrated probability of future disease.')
        st.subheader('Input values')
        view = input_df.copy(); view['_BMI5'] = view['_BMI5'] / 100
        view = view.rename(columns={'_AGE80':'Age', '_SEX':'Sex code', '_BMI5':'BMI', 'BPHIGH6':'BP code', 'DIABETE4':'Diabetes code', '_SMOKER3':'Smoking code', 'GENHLTH':'General health code'})
        st.dataframe(view, use_container_width=True)
        cur.execute('''INSERT INTO predictions (age, sex, bmi, blood_pressure, diabetes, smoker, general_health, predicted_class, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', (age, sex_code, bmi, bp_code, diabetes_code, smoker_code, health_code, pred, datetime.now().isoformat()))
        conn.commit()
        st.success('Prediction saved in the database.')
    except Exception as e:
        st.error('Prediction could not be completed. Error details: ' + str(e))

st.header('4. Prediction History')
history = pd.read_sql_query('SELECT * FROM predictions ORDER BY id DESC', conn)
if history.empty:
    st.info('No predictions yet. Run your first prediction above.')
else:
    st.dataframe(history, use_container_width=True)
conn.close()
st.caption('CardioLens | Student project prototype')
