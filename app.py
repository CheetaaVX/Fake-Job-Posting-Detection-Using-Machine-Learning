import streamlit as st
import joblib

st.set_page_config(page_title="Fake Job Detection", layout="wide")

model = joblib.load("fake_job_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.markdown("""
<style>
/* Base */
body {
    background: #0f1115;
    color: #e4e6eb;
}

/* Layout */
.block-container {
    padding: 1.5rem 3rem;
    max-width: 1100px;
}

/* Title spacing fix */
h1 {
    margin-bottom: 5px !important;
}

p {
    margin-top: 0px !important;
    margin-bottom: 12px !important;
}

/* Labels */
label {
    font-size: 14px !important;
    color: #aab0b6 !important;
}

/* Inputs */
.stTextInput input,
.stTextArea textarea {
    background: #0f1115;
    color: #e4e6eb;
    border: 1px solid #2a2f38;
    border-radius: 6px;
    padding: 8px;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border: 1px solid #3b82f6;
    outline: none;
}

/* Button */
.stButton button {
    background: #3b82f6;
    color: white;
    border-radius: 6px;
    height: 40px;
    width: 100%;
    font-weight: 500;
}

.stButton button:hover {
    background: #2563eb;
}

/* Result Box */
.result {
    padding: 14px;
    border-radius: 8px;
    text-align: center;
    font-weight: 600;
    margin-top: 10px;
}

.fake {
    background: #2a1414;
    color: #ff6b6b;
}

.real {
    background: #132a1a;
    color: #4ade80;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1>Fake Job Detection</h1>
<p style='color:#aab0b6;'>Check whether a job post is real or fake using Machine Learning</p>
""", unsafe_allow_html=True)

left, right = st.columns([2, 1])

with left:
    st.subheader("Basic Info")

    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Job title")
    with col2:
        location = st.text_input("Location")

    department = st.text_input("Department")

    st.subheader("Details")

    company_profile = st.text_area("Company profile", height=100)
    description = st.text_area("Description", height=150)
    requirements = st.text_area("Requirements", height=120)
    benefits = st.text_area("Benefits", height=100)

    st.write("")
    run = st.button("Analyze")

with right:
    st.subheader("Result")

    if run:
        text = " ".join([
            str(title),
            str(location),
            str(department),
            str(company_profile),
            str(description),
            str(requirements),
            str(benefits)
        ])

        X = vectorizer.transform([text])
        pred = model.predict(X)[0]

        if pred == 1:
            label, cls = "Fake Job", "fake"
        else:
            label, cls = "Real Job", "real"

        # Probability 
        try:
            prob = model.predict_proba(X)[0][pred]
            label = f"{label} ({prob:.2f})"
        except:
            pass

        st.markdown(f'<div class="result {cls}">{label}</div>', unsafe_allow_html=True)

    else:
        st.write("Waiting for input...")