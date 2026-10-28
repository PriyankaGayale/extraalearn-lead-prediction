import streamlit as st
import joblib
import pandas as pd

# Load the trained model pipeline
model = joblib.load("extraalearn_model.pkl")

# Set up the Streamlit page title and description
st.title("ExtraaLearn Lead Conversion Predictor")
st.write("Enter the lead's details below to predict their conversion probability.")

# Create input widgets for the user
# (Adjust these features to match the actual columns of your ExtraaLearn dataset)
course_enrolled = st.selectbox("Course Enrolled", ["No", "Yes"])
time_spent = st.number_input("Time Spent on Website (minutes)", min_value=0.0, value=10.0)
profile_completed = st.selectbox("Profile Completed", ["Low", "Medium", "High"])
website_visits = st.number_input("Website Visits", min_value=0, value=2)
last_activity = st.selectbox("Last Activity", ["Content Page", "Email Link", "Phone Call", "Website Visit"])

# When the user clicks the 'Predict' button
if st.button("Predict Lead Conversion"):
    # Store inputs in a DataFrame matching the model's expected features
    input_data = pd.DataFrame({
        "Course_Enrolled": [course_enrolled],
        "Time_Spent_on_Website": [time_spent],
        "Profile_Completed": [profile_completed],
        "Website_Visits": [website_visits],
        "Last_Activity": [last_activity]
    })

    # Make prediction and get probability
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display the result to the user
    if prediction == 1:
        st.success(f"High Intent Lead! (Likely to Convert)\n\n **Conversion Probability:** {probability:.2f}")
    else:
        st.warning(f"Low Intent Lead. (Unlikely to Convert)\n\n **Conversion Probability:** {probability:.2f}")
