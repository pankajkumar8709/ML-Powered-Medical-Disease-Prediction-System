import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time
import plotly.express as px
from datetime import datetime

# Cache model and chatbot loading
@st.cache_data
def load_models():
    try:
        model_diabetes = joblib.load('D:\\Code Base\\Code Base\\Streamlit\\eday\\model_diabetes_optimized.pkl')
        model_heart = joblib.load('D:\\Code Base\\Code Base\\Streamlit\\eday\\model_heart_optimized.pkl')
        model_kidney = joblib.load('D:\\Code Base\\Code Base\\Streamlit\\eday\\model_kidney_optimized.pkl')
        scaler_diabetes = joblib.load('D:\\Code Base\\Code Base\\Streamlit\\eday\\scaler_diabetes.pkl')
        scaler_heart = joblib.load('D:\\Code Base\\Code Base\\Streamlit\\eday\\scaler_heart.pkl')
        scaler_kidney = joblib.load('D:\\Code Base\\Code Base\\Streamlit\\eday\\scaler_kidney.pkl')
        st.sidebar.success("✅ Using optimized models with scalers.")
        return model_diabetes, model_heart, model_kidney, scaler_diabetes, scaler_heart, scaler_kidney
    except FileNotFoundError:
        st.sidebar.warning("⚠️ Optimized models/scalers not found. Falling back to basic models.")
        model_diabetes = joblib.load('D:\\Code Base\\Code Base\\Streamlit\\eday\\model_diabetes.pkl')
        model_heart = joblib.load('D:\\Code Base\\Code Base\\Streamlit\\eday\\model_heart.pkl')
        model_kidney = joblib.load('D:\\Code Base\\Code Base\\Streamlit\\eday\\model_kidney.pkl')
        X_dummy_diabetes, _ = make_classification(n_samples=100, n_features=8, random_state=42)
        X_dummy_heart, _ = make_classification(n_samples=100, n_features=15, random_state=42)
        X_dummy_kidney, _ = make_classification(n_samples=100, n_features=42, random_state=42)
        scaler_diabetes = StandardScaler().fit(pd.DataFrame(X_dummy_diabetes, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']))
        scaler_heart = StandardScaler().fit(pd.DataFrame(X_dummy_heart, columns=['id', 'age', 'sex', 'dataset', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']))
        scaler_kidney = StandardScaler().fit(pd.DataFrame(X_dummy_kidney, columns=['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'rbg', 'bu', 'sc', 'sodium', 'potassium', 'hb', 'pcv', 'wc', 'rc',
                                                                                 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'egfr', 'upcr', 'uo', 'salb', 'chol', 'pth', 'calcium', 'phosphate',
                                                                                 'fh', 'smoking', 'bmi', 'pal', 'ddm', 'dhtn', 'cystatin', 'usm', 'crp', 'il6']))
        st.sidebar.warning("⚠️ Fallback scalers fitted with dummy data. Results may be inaccurate.")
        return model_diabetes, model_heart, model_kidney, scaler_diabetes, scaler_heart, scaler_kidney

@st.cache_data
def load_chatbot():
    model_name = "distilgpt2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=200, temperature=0.7)
        st.success("✅ Chatbot loaded successfully.")
        return pipe
    except Exception as e:
        st.error(f"❌ Failed to load chatbot: {e}. Chatbot disabled.")
        return None

# Load models and chatbot
model_diabetes, model_heart, model_kidney, scaler_diabetes, scaler_heart, scaler_kidney = load_models()
chatbot = load_chatbot()

# Title with custom styling
st.markdown("""
    <style>
    .title {
        color: #2c3e50;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        padding: 10px;
        background-color: #ecf0f1;
        border-radius: 5px;
    }
    </style>
    <div class="title">Medical Disease Prediction App</div>
""", unsafe_allow_html=True)

# Sidebar with dataset selection and feedback
st.sidebar.header("Settings")
dataset = st.sidebar.selectbox("Select Type of Disease", ["Diabetes", "Heart", "Kidney"])
feedback = st.sidebar.text_area("Your Feedback", help="Let us know how we can improve!")
if st.sidebar.button("Submit Feedback"):
    with open("feedback.txt", "a") as f:
        f.write(f"{datetime.now()}: {feedback}\n")
    st.sidebar.success("Thank you for your feedback!")

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'latest_risk_score' not in st.session_state:
    st.session_state.latest_risk_score = None

def check_features(model, X):
    try:
        expected = model.n_features_in_
        if X.shape[1] != expected:
            st.error(f"⚠️ Model expects {expected} features, but got {X.shape[1]}. Check input order/columns!")
            return False
    except AttributeError:
        pass
    return True

# Diabetes
if dataset == "Diabetes":
    st.header("Diabetes Prediction")
    with st.expander("Enter Key Inputs", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            glucose = st.number_input("Glucose (mg/dL) *", min_value=0, max_value=200, value=120, help="Blood glucose level in mg/dL")
            if glucose > 200: st.warning("Glucose exceeds 200 mg/dL, which may be unrealistic!")
            age = st.number_input("Age *", min_value=21, max_value=81, value=30, help="Age in years")
            if age > 81: st.warning("Age exceeds 81, which may be unrealistic!")
        with col2:
            bmi = st.number_input("BMI *", min_value=0.0, max_value=67.1, value=25.0, help="Body Mass Index")
            if bmi > 67.1: st.warning("BMI exceeds 67.1, which may be unrealistic!")
            pedigree = st.number_input("Diabetes Pedigree Function *", min_value=0.0, max_value=2.42, value=0.5, help="Diabetes predisposition score")
            if pedigree > 2.42: st.warning("Pedigree function exceeds 2.42, which may be unrealistic!")
        optional = st.expander("Optional Inputs")
        with optional:
            col3, col4 = st.columns(2)
            with col3:
                pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=0, help="Number of pregnancies")
                if pregnancies > 17: st.warning("Pregnancies exceed 17, which may be unrealistic!")
                blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=122, value=70, help="Blood pressure in mm Hg")
                if blood_pressure > 122: st.warning("Blood pressure exceeds 122 mm Hg, which may be unrealistic!")
            with col4:
                skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=99, value=20, help="Skin thickness in mm")
                if skin_thickness > 99: st.warning("Skin thickness exceeds 99 mm, which may be unrealistic!")
                insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=846, value=79, help="Insulin level in mu U/ml")
                if insulin > 846: st.warning("Insulin exceeds 846 mu U/ml, which may be unrealistic!")

    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]],
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    input_data_scaled = scaler_diabetes.transform(input_data)
    if st.button("Predict", key="predict_diabetes"):
        with st.spinner("Analyzing..."):
            time.sleep(1)
            if check_features(model_diabetes, input_data_scaled):
                prediction = model_diabetes.predict(input_data_scaled)
                st.session_state.latest_risk_score = prediction[0]
                risk_score = st.session_state.latest_risk_score
                score = int((1 - risk_score) * 100)
                st.metric("Your Health Score", f"{score}/100")
                if risk_score == 0:
                    st.success("Prediction: Not Diabetic (Score: 0)")
                    if chatbot:
                        with st.spinner("Generating advice..."):
                            chatbot_response = chatbot("What advice would you give for someone not diabetic?", return_full_text=False)
                            if chatbot_response and len(chatbot_response) > 0 and 'generated_text' in chatbot_response[0]:
                                st.write("Chatbot: ", chatbot_response[0]['generated_text'])
                            else:
                                st.write("Chatbot: Sorry, I couldn't generate a response.")
                else:
                    st.error(f"Prediction: Diabetic Risk (Score: {risk_score})")
                    if chatbot:
                        with st.spinner("Generating advice..."):
                            chatbot_response = chatbot(f"What dietary advice would you give for a diabetic with glucose {glucose} mg/dL and BMI {bmi}?", return_full_text=False)
                            if chatbot_response and len(chatbot_response) > 0 and 'generated_text' in chatbot_response[0]:
                                st.write("Chatbot: ", chatbot_response[0]['generated_text'])
                            else:
                                st.write("Chatbot: Sorry, I couldn't generate a response.")

                fig = px.bar(x=['Glucose', 'BMI', 'Age'], y=[glucose, bmi, age], title="Key Input Values")
                st.plotly_chart(fig)

                risk_factors = []
                if glucose > 120: risk_factors.append(f"High glucose ({glucose} mg/dL)")
                if bmi > 30: risk_factors.append(f"High BMI ({bmi})")
                if age > 50: risk_factors.append("Advanced age (>50 years)")
                if risk_factors:
                    st.write("Risk Factors:", ", ".join(risk_factors))
                else:
                    st.write("No significant risk factors detected.")

    col5, col6, col7 = st.columns(3)
    with col5:
        if st.button("Save Prediction", key="save_diabetes"):
            if st.session_state.latest_risk_score is not None:
                st.session_state.prediction_history.append({
                    "inputs": input_data.iloc[0].tolist(),
                    "score": st.session_state.latest_risk_score,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.success("Prediction saved to history!")
            else:
                st.warning("Please predict first to save a score!")
    with col6:
        if st.button("Show Risk Score History", key="history_diabetes"):
            if st.session_state.prediction_history:
                chart_data = pd.DataFrame(st.session_state.prediction_history)
                chart_data['time'] = pd.to_datetime(chart_data['time'])
                st.line_chart(chart_data.set_index('time')['score'])
            else:
                st.write("No prediction history available.")
    with col7:
        if st.button("Export Prediction", key="export_diabetes"):
            if st.session_state.prediction_history:
                latest = st.session_state.prediction_history[-1]
                export_data = pd.DataFrame([latest['inputs'] + [latest['score']]],
                                          columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'risk_score'])
                csv = export_data.to_csv(index=False)
                st.download_button(label="Download CSV", data=csv, file_name="diabetes_prediction.csv", mime="text/csv")
            else:
                st.write("No prediction to export. Predict first!")

# Heart
elif dataset == "Heart":
    st.header("Heart Disease Prediction")
    with st.expander("Enter Key Inputs", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age *", min_value=0, max_value=100, value=50, help="Age in years")
            if age > 100: st.warning("Age exceeds 100, which may be unrealistic!")
            cp = st.selectbox("Chest Pain Type (0-3) *", [0, 1, 2, 3], index=0, help="Type of chest pain")
        with col2:
            thalch = st.number_input("Max Heart Rate *", min_value=0, max_value=202, value=150, help="Maximum heart rate in bpm")
            if thalch > 202 or thalch < 30: st.warning("Max heart rate should be 30–202 bpm!")
            oldpeak = st.number_input("Oldpeak *", min_value=0.0, max_value=6.2, value=1.0, help="ST depression induced by exercise")
            if oldpeak > 6.2: st.warning("Oldpeak exceeds 6.2, which may be unrealistic!")
        optional = st.expander("Optional Inputs")
        with optional:
            col3, col4 = st.columns(2)
            with col3:
                sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1], index=0, help="Gender")
                trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=200, value=120, help="Resting BP in mm Hg")
                if trestbps > 200: st.warning("Resting BP exceeds 200 mm Hg, which may be unrealistic!")
            with col4:
                chol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=200, help="Cholesterol level in mg/dl")
                if chol > 600: st.warning("Cholesterol exceeds 600 mg/dl, which may be unrealistic!")
                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], index=0, help="Fasting blood sugar > 120 mg/dl")

    input_data = pd.DataFrame([[1, age, sex, 0, cp, trestbps, chol, fbs, 0, thalch, 0, oldpeak, 0, 0, 0]],
                              columns=['id', 'age', 'sex', 'dataset', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    input_data_scaled = scaler_heart.transform(input_data)
    if st.button("Predict", key="predict_heart"):
        with st.spinner("Analyzing..."):
            time.sleep(1)
            if check_features(model_heart, input_data_scaled):
                prediction = model_heart.predict(input_data_scaled)
                st.session_state.latest_risk_score = prediction[0]
                risk_score = st.session_state.latest_risk_score
                score = int((1 - risk_score) * 100)
                st.metric("Your Health Score", f"{score}/100")
                if risk_score == 0:
                    st.success("Prediction: No Heart Disease Risk (Score: 0)")
                    if chatbot:
                        with st.spinner("Generating advice..."):
                            chatbot_response = chatbot("What lifestyle advice would you give for someone with no heart disease risk?", return_full_text=False)
                            if chatbot_response and len(chatbot_response) > 0 and 'generated_text' in chatbot_response[0]:
                                st.write("Chatbot: ", chatbot_response[0]['generated_text'])
                            else:
                                st.write("Chatbot: Sorry, I couldn't generate a response.")
                elif 1 <= risk_score <= 2:
                    st.warning(f"Prediction: Mild to Moderate Heart Disease Risk (Score: {risk_score})")
                    if chatbot:
                        with st.spinner("Generating advice..."):
                            chatbot_response = chatbot(f"What lifestyle changes would help someone with a heart risk score of {risk_score} and cholesterol {chol} mg/dl?", return_full_text=False)
                            if chatbot_response and len(chatbot_response) > 0 and 'generated_text' in chatbot_response[0]:
                                st.write("Chatbot: ", chatbot_response[0]['generated_text'])
                            else:
                                st.write("Chatbot: Sorry, I couldn't generate a response.")
                elif risk_score >= 3:
                    st.error(f"Prediction: High Heart Disease Risk (Score: {risk_score})")
                    if chatbot:
                        with st.spinner("Generating advice..."):
                            chatbot_response = chatbot(f"What immediate steps should someone with a high heart risk score of {risk_score} and oldpeak {oldpeak} take?", return_full_text=False)
                            if chatbot_response and len(chatbot_response) > 0 and 'generated_text' in chatbot_response[0]:
                                st.write("Chatbot: ", chatbot_response[0]['generated_text'])
                            else:
                                st.write("Chatbot: Sorry, I couldn't generate a response.")

                fig = px.bar(x=['Age', 'Cholesterol', 'Oldpeak'], y=[age, chol, oldpeak], title="Key Input Values")
                st.plotly_chart(fig)

                risk_factors = []
                if age > 60: risk_factors.append("Advanced age (>60 years)")
                if cp > 1: risk_factors.append("Significant chest pain type")
                if chol > 200: risk_factors.append(f"High cholesterol ({chol} mg/dl)")
                if oldpeak > 1.0: risk_factors.append(f"Elevated ST depression ({oldpeak})")
                if risk_factors:
                    st.write("Risk Factors:", ", ".join(risk_factors))
                else:
                    st.write("No significant risk factors detected.")

    col5, col6, col7 = st.columns(3)
    with col5:
        if st.button("Save Prediction", key="save_heart"):
            if st.session_state.latest_risk_score is not None:
                st.session_state.prediction_history.append({
                    "inputs": input_data.iloc[0].tolist(),
                    "score": st.session_state.latest_risk_score,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.success("Prediction saved to history!")
            else:
                st.warning("Please predict first to save a score!")
    with col6:
        if st.button("Show Risk Score History", key="history_heart"):
            if st.session_state.prediction_history:
                chart_data = pd.DataFrame(st.session_state.prediction_history)
                chart_data['time'] = pd.to_datetime(chart_data['time'])
                st.line_chart(chart_data.set_index('time')['score'])
            else:
                st.write("No prediction history available.")
    with col7:
        if st.button("Export Prediction", key="export_heart"):
            if st.session_state.prediction_history:
                latest = st.session_state.prediction_history[-1]
                export_data = pd.DataFrame([latest['inputs'] + [latest['score']]],
                                          columns=['id', 'age', 'sex', 'dataset', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'risk_score'])
                csv = export_data.to_csv(index=False)
                st.download_button(label="Download CSV", data=csv, file_name="heart_prediction.csv", mime="text/csv")
            else:
                st.write("No prediction to export. Predict first!")

# Kidney
elif dataset == "Kidney":
    st.header("Kidney Disease Prediction")
    with st.expander("Enter Key Inputs", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age *", min_value=0, max_value=120, value=50, help="Age in years")
            if age > 120: st.warning("Age exceeds 120 years, which may be unrealistic!")
            bu = st.number_input("Blood Urea (mg/dl) *", min_value=0.0, max_value=200.0, value=20.0, help="Blood urea in mg/dl")
            if bu > 200: st.warning("Blood urea exceeds 200 mg/dl, which may be unrealistic!")
        with col2:
            sc = st.number_input("Serum Creatinine (mg/dl) *", min_value=0.0, max_value=20.0, value=1.0, help="Serum creatinine in mg/dl")
            if sc > 20: st.warning("Serum creatinine exceeds 20 mg/dl, which may be unrealistic!")
            il6 = st.number_input("IL-6 Level *", min_value=0.0, max_value=100.0, value=2.0, help="Interleukin-6 level")
            if il6 > 100: st.warning("IL-6 level exceeds 100, which may be unrealistic!")
        optional = st.expander("Optional Inputs")
        with optional:
            col3, col4 = st.columns(2)
            with col3:
                sodium = st.number_input("Sodium (mEq/L)", min_value=0.0, max_value=150.0, value=140.0, help="Sodium level in mEq/L")
                if sodium < 130 or sodium > 150: st.warning("Sodium level should be 130–150 mEq/L!")
                hb = st.number_input("Hemoglobin (gms)", min_value=0.0, max_value=20.0, value=13.0, help="Hemoglobin in gms")
                if hb > 20: st.warning("Hemoglobin exceeds 20 gms, which may be unrealistic!")
            with col4:
                potassium = st.number_input("Potassium (mEq/L)", min_value=0.0, max_value=10.0, value=4.0, help="Potassium level in mEq/L")
                if potassium > 10: st.warning("Potassium exceeds 10 mEq/L, which may be unrealistic!")
                pcv = st.number_input("Packed Cell Volume (%)", min_value=0, max_value=100, value=40, help="Packed cell volume in %")
                if pcv > 100: st.warning("Packed cell volume exceeds 100%, which may be unrealistic!")
                dhtn = st.number_input("Duration of Hypertension (years)", min_value=0, max_value=50, value=0, help="Duration in years")
                if dhtn > 50: st.warning("Duration of hypertension exceeds 50 years, which may be unrealistic!")

    # Updated input_data with matching feature names
    input_data = pd.DataFrame([[age, 80, 1.01, 0, 0, 0, 0, 0, 0, 100, bu, sc, sodium, potassium, hb, pcv, 7000, 4.5,
                               0, 0, 0, 0, 0, 0, 90.0, 0.1, 1500, 4.0, 200, 50.0, 9.0, 3.5,
                               0, 0, 25.0, 0, 0, dhtn, 1.0, 0, 5.0, il6]],
                              columns=['Age of the patient', 'Blood pressure (mm/Hg)', 'Specific gravity of urine', 'Albumin in urine', 'Sugar in urine', 'Red blood cells in urine', 'Pus cells in urine', 'Pus cell clumps in urine', 'Bacteria in urine', 'Random blood glucose level (mg/dl)', 'Blood urea (mg/dl)', 'Serum creatinine (mg/dl)', 'Sodium level (mEq/L)', 'Potassium level (mEq/L)', 'Hemoglobin level (gms)', 'Packed cell volume (%)', 'White blood cell count (cells/cumm)', 'Red blood cell count (millions/cumm)',
                                       'Hypertension (yes/no)', 'Diabetes mellitus (yes/no)', 'Coronary artery disease (yes/no)', 'Appetite (good/poor)', 'Pedal edema (yes/no)', 'Anemia (yes/no)', 'Estimated Glomerular Filtration Rate (eGFR)', 'Urine protein-to-creatinine ratio', 'Urine output (ml/day)', 'Serum albumin level', 'Cholesterol level', 'Parathyroid hormone (PTH) level', 'Serum calcium level', 'Serum phosphate level',
                                       'Family history of chronic kidney disease', 'Smoking status', 'Body Mass Index (BMI)', 'Physical activity level', 'Duration of diabetes mellitus (years)', 'Duration of hypertension (years)', 'Cystatin C level', 'Urinary sediment microscopy results', 'C-reactive protein (CRP) level', 'Interleukin-6 (IL-6) level'])
    input_data_scaled = scaler_kidney.transform(input_data)
    if st.button("Predict", key="predict_kidney"):
        with st.spinner("Analyzing..."):
            time.sleep(1)
            if check_features(model_kidney, input_data_scaled):
                prediction = model_kidney.predict(input_data_scaled)
                st.session_state.latest_risk_score = prediction[0]
                risk_score = st.session_state.latest_risk_score
                score = int((1 - risk_score) * 100)
                st.metric("Your Health Score", f"{score}/100")
                if risk_score == 0:
                    st.success("Prediction: No Kidney Disease (Score: 0)")
                    if chatbot:
                        with st.spinner("Generating advice..."):
                            chatbot_response = chatbot("What advice would you give for someone with no kidney disease risk?", return_full_text=False)
                            if chatbot_response and len(chatbot_response) > 0 and 'generated_text' in chatbot_response[0]:
                                st.write("Chatbot: ", chatbot_response[0]['generated_text'])
                            else:
                                st.write("Chatbot: Sorry, I couldn't generate a response.")
                else:
                    st.error(f"Prediction: Kidney Disease Risk (Score: {risk_score})")
                    if chatbot:
                        with st.spinner("Generating advice..."):
                            chatbot_response = chatbot(f"What kidney health tips would you suggest for someone with a risk score of {risk_score} and blood urea {bu} mg/dl?", return_full_text=False)
                            if chatbot_response and len(chatbot_response) > 0 and 'generated_text' in chatbot_response[0]:
                                st.write("Chatbot: ", chatbot_response[0]['generated_text'])
                            else:
                                st.write("Chatbot: Sorry, I couldn't generate a response.")

                fig = px.bar(x=['Blood Urea', 'Serum Creatinine', 'Sodium'], y=[bu, sc, sodium], title="Key Input Values")
                st.plotly_chart(fig)

                risk_factors = []
                if bu > 50: risk_factors.append(f"High blood urea ({bu} mg/dl)")
                if sc > 2.0: risk_factors.append(f"High serum creatinine ({sc} mg/dl)")
                if sodium < 130 or sodium > 150: risk_factors.append(f"Abnormal sodium ({sodium} mEq/L)")
                if potassium > 5.0: risk_factors.append(f"High potassium ({potassium} mEq/L)")
                if risk_factors:
                    st.write("Risk Factors:", ", ".join(risk_factors))
                else:
                    st.write("No significant risk factors detected.")

    col5, col6, col7 = st.columns(3)
    with col5:
        if st.button("Save Prediction", key="save_kidney"):
            if st.session_state.latest_risk_score is not None:
                st.session_state.prediction_history.append({
                    "inputs": input_data.iloc[0].tolist(),
                    "score": st.session_state.latest_risk_score,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.success("Prediction saved to history!")
            else:
                st.warning("Please predict first to save a score!")
    with col6:
        if st.button("Show Risk Score History", key="history_kidney"):
            if st.session_state.prediction_history:
                chart_data = pd.DataFrame(st.session_state.prediction_history)
                chart_data['time'] = pd.to_datetime(chart_data['time'])
                st.line_chart(chart_data.set_index('time')['score'])
            else:
                st.write("No prediction history available.")
    with col7:
        if st.button("Export Prediction", key="export_kidney"):
            if st.session_state.prediction_history:
                latest = st.session_state.prediction_history[-1]
                export_data = pd.DataFrame([latest['inputs'] + [latest['score']]],
                                          columns=['Age of the patient', 'Blood pressure (mm/Hg)', 'Specific gravity of urine', 'Albumin in urine', 'Sugar in urine', 'Red blood cells in urine', 'Pus cells in urine', 'Pus cell clumps in urine', 'Bacteria in urine', 'Random blood glucose level (mg/dl)', 'Blood urea (mg/dl)', 'Serum creatinine (mg/dl)', 'Sodium level (mEq/L)', 'Potassium level (mEq/L)', 'Hemoglobin level (gms)', 'Packed cell volume (%)', 'White blood cell count (cells/cumm)', 'Red blood cell count (millions/cumm)',
                                                  'Hypertension (yes/no)', 'Diabetes mellitus (yes/no)', 'Coronary artery disease (yes/no)', 'Appetite (good/poor)', 'Pedal edema (yes/no)', 'Anemia (yes/no)', 'Estimated Glomerular Filtration Rate (eGFR)', 'Urine protein-to-creatinine ratio', 'Urine output (ml/day)', 'Serum albumin level', 'Cholesterol level', 'Parathyroid hormone (PTH) level', 'Serum calcium level', 'Serum phosphate level',
                                                  'Family history of chronic kidney disease', 'Smoking status', 'Body Mass Index (BMI)', 'Physical activity level', 'Duration of diabetes mellitus (years)', 'Duration of hypertension (years)', 'Cystatin C level', 'Urinary sediment microscopy results', 'C-reactive protein (CRP) level', 'Interleukin-6 (IL-6) level', 'risk_score'])
                csv = export_data.to_csv(index=False)
                st.download_button(label="Download CSV", data=csv, file_name="kidney_prediction.csv", mime="text/csv")
            else:
                st.write("No prediction to export. Predict first!")

# Footer
st.sidebar.info("👉 Enter key inputs (marked with *) and use additional features to explore predictions.")