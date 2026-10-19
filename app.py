import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# --------------------------------------------------
# 2. LOAD THE TRAINED MACHINE LEARNING MODEL
# --------------------------------------------------

MODEL_PATH = Path(__file__).parent / "student_performance_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(
        "The trained model file was not found. "
        "Please ensure student_performance_model.pkl "
        "is in the same folder as app.py."
    )
    st.stop()

# --------------------------------------------------
# 3. APPLICATION TITLE
# --------------------------------------------------

st.title("🎓 Student Performance Prediction System")

st.write(
    """
    Welcome to the Student Performance Prediction System!

    This application uses a trained Machine Learning model
    to estimate whether a student is likely to pass or may
    need additional academic support.
    """
)

st.info(
    "This tool provides an estimate, not a guarantee of "
    "academic performance. Predictions should support "
    "students, not determine their opportunities."
)

st.divider()

# --------------------------------------------------
# 4. STUDENT INFORMATION
# --------------------------------------------------

st.header("📋 Student Information")

st.subheader("School and Personal Details")

col1, col2 = st.columns(2)

with col1:
    school = st.selectbox(
        "School",
        ["GP", "MS"]
    )

    sex = st.selectbox(
        "Sex",
        ["F", "M"]
    )

    age = st.slider(
        "Age",
        min_value=15,
        max_value=22,
        value=17
    )

    address = st.selectbox(
        "Home Address Type",
        ["U", "R"],
        help="U = Urban, R = Rural"
    )

with col2:
    famsize = st.selectbox(
        "Family Size",
        ["GT3", "LE3"],
        help="GT3 = More than 3, LE3 = 3 or fewer"
    )

    Pstatus = st.selectbox(
        "Parents' Cohabitation Status",
        ["T", "A"],
        help="T = Living together, A = Apart"
    )

    Medu = st.selectbox(
        "Mother's Education Level",
        [0, 1, 2, 3, 4],
        help="0 = None, 1 = Primary, 2 = 5th–9th grade, "
             "3 = Secondary, 4 = Higher education"
    )

    Fedu = st.selectbox(
        "Father's Education Level",
        [0, 1, 2, 3, 4],
        help="0 = None, 1 = Primary, 2 = 5th–9th grade, "
             "3 = Secondary, 4 = Higher education"
    )

st.divider()

# --------------------------------------------------
# 5. ACADEMIC INFORMATION
# --------------------------------------------------

st.subheader("📚 Academic Information")

col3, col4 = st.columns(2)

with col3:
    studytime = st.selectbox(
        "Weekly Study Time",
        [1, 2, 3, 4],
        help="1 = Less than 2 hours, "
             "2 = 2–5 hours, "
             "3 = 5–10 hours, "
             "4 = More than 10 hours"
    )

    failures = st.selectbox(
        "Number of Previous Class Failures",
        [0, 1, 2, 3],
        help="Select the number of past class failures."
    )

    G1 = st.slider(
        "First Period Grade (G1)",
        min_value=0,
        max_value=20,
        value=10,
        help="Grade from 0 to 20."
    )

with col4:
    absences = st.number_input(
        "Number of School Absences",
        min_value=0,
        max_value=100,
        value=0,
        step=1
    )

    schoolsup = st.selectbox(
        "Extra Educational Support",
        ["yes", "no"]
    )

    famsup = st.selectbox(
        "Family Educational Support",
        ["yes", "no"]
    )

paid = st.selectbox(
    "Extra Paid Classes",
    ["yes", "no"]
)

st.divider()

# --------------------------------------------------
# 6. FAMILY AND SOCIAL INFORMATION
# --------------------------------------------------

st.subheader("🏠 Family and Social Information")

col5, col6 = st.columns(2)

with col5:
    activities = st.selectbox(
        "Extracurricular Activities",
        ["yes", "no"]
    )

    nursery = st.selectbox(
        "Attended Nursery School",
        ["yes", "no"]
    )

    higher = st.selectbox(
        "Wants to Pursue Higher Education",
        ["yes", "no"]
    )

    internet = st.selectbox(
        "Internet Access at Home",
        ["yes", "no"]
    )

    romantic = st.selectbox(
        "Currently in a Romantic Relationship",
        ["yes", "no"]
    )

with col6:
    famrel = st.slider(
        "Quality of Family Relationships",
        min_value=1,
        max_value=5,
        value=4,
        help="1 = Very bad, 5 = Excellent"
    )

    freetime = st.slider(
        "Free Time After School",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Very low, 5 = Very high"
    )

    goout = st.slider(
        "Going Out with Friends",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Very low, 5 = Very high"
    )

    Dalc = st.slider(
        "Workday Alcohol Consumption",
        min_value=1,
        max_value=5,
        value=1,
        help="1 = Very low, 5 = Very high"
    )

    Walc = st.slider(
        "Weekend Alcohol Consumption",
        min_value=1,
        max_value=5,
        value=1,
        help="1 = Very low, 5 = Very high"
    )

health = st.slider(
    "Current Health Status",
    min_value=1,
    max_value=5,
    value=4,
    help="1 = Very poor, 5 = Very good"
)

st.divider()

# --------------------------------------------------
# 7. PREPARE INPUT DATA
# --------------------------------------------------

# These five fields are not entered by the user in this
# version of the application. Default values are supplied.
# The model expects these features because they were
# included in the original training dataset.

input_data = pd.DataFrame([{
    "school": school,
    "sex": sex,
    "age": age,
    "address": address,
    "famsize": famsize,
    "Pstatus": Pstatus,
    "Medu": Medu,
    "Fedu": Fedu,
    "Mjob": "other",
    "Fjob": "other",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 1,
    "studytime": studytime,
    "failures": failures,
    "schoolsup": schoolsup,
    "famsup": famsup,
    "paid": paid,
    "activities": activities,
    "nursery": nursery,
    "higher": higher,
    "internet": internet,
    "romantic": romantic,
    "famrel": famrel,
    "freetime": freetime,
    "goout": goout,
    "Dalc": Dalc,
    "Walc": Walc,
    "health": health,
    "absences": absences,
    "G1": G1
}])

# --------------------------------------------------
# 8. MAKE THE PREDICTION
# --------------------------------------------------

st.header("🔍 Predict Student Performance")

st.write(
    "Click the button below to estimate the student's "
    "academic outcome using the trained model."
)

if st.button("Predict Student Outcome", type="primary"):

    try:
        # Predict the outcome
        prediction = model.predict(input_data)[0]

        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.success(
                "The model predicts that the student "
                "is likely to pass."
            )
        else:
            st.warning(
                "The model identifies this student as "
                "potentially at risk of failing. Consider "
                "reviewing their individual circumstances "
                "and offering appropriate academic support."
            )

        # Display estimated probabilities when available
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            # Find the probability corresponding to class 1
            classes = list(model.classes_)
            pass_index = classes.index(1)

            pass_probability = probabilities[pass_index]

            st.metric(
                label="Estimated Probability of Passing",
                value=f"{pass_probability:.1%}"
            )

            st.progress(float(pass_probability))

            st.caption(
                "This probability is a model estimate, "
                "not a guarantee of the student's final result."
            )

    except Exception as e:
        st.error(
            "An error occurred while making the prediction."
        )

        st.exception(e)

# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Student Performance Prediction System | "
    "Machine Learning Academic Project"
)