"""
ExtraaLearn Lead Conversion Predictor — Streamlit Frontend
============================================================
A thin UI layer that collects raw lead attributes from the user and sends
them to the Flask backend's /predict endpoint. This app does NOT load the
trained model directly — the Flask API (backed by the serialized sklearn
Pipeline) remains the single source of inference truth.
"""

import os

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Backend configuration — configurable via environment variable, never
# hardcoded to a specific cloud URL.
# ---------------------------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000/predict")

st.set_page_config(
    page_title="ExtraaLearn Lead Conversion Predictor",
    page_icon="\U0001F4C8",
    layout="centered",
)

st.title("ExtraaLearn Lead Conversion Predictor")
st.write(
    "Estimate the probability that a lead will convert into a paid customer, "
    "based on their profile and engagement with ExtraaLearn. This tool is "
    "designed to help the sales team prioritize outreach."
)

st.divider()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.subheader("Lead Profile")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=63, value=51)
    current_occupation = st.selectbox(
        "Current Occupation", ["Professional", "Unemployed", "Student"]
    )
    first_interaction = st.selectbox("First Interaction", ["Website", "Mobile App"])
    profile_completed = st.selectbox("Profile Completed", ["High", "Medium", "Low"])

with col2:
    last_activity = st.selectbox(
        "Last Activity", ["Email Activity", "Phone Activity", "Website Activity"]
    )
    website_visits = st.number_input(
        "Website Visits", min_value=0, max_value=30, value=3, step=1
    )
    time_spent_on_website = st.number_input(
        "Time Spent on Website (seconds)", min_value=0, max_value=2537, value=376, step=1
    )
    page_views_per_visit = st.number_input(
        "Page Views per Visit", min_value=0.0, max_value=18.5, value=2.79, step=0.1, format="%.2f"
    )

st.subheader("Marketing & Referral Exposure")
col3, col4, col5 = st.columns(3)

with col3:
    print_media_type1 = st.radio("Seen in Newspaper?", ["No", "Yes"], horizontal=True)
    print_media_type2 = st.radio("Seen in Magazine?", ["No", "Yes"], horizontal=True)

with col4:
    digital_media = st.radio("Seen on Digital Media?", ["No", "Yes"], horizontal=True)
    educational_channels = st.radio("Heard via Educational Channels?", ["No", "Yes"], horizontal=True)

with col5:
    referral = st.radio("Came via Referral?", ["No", "Yes"], horizontal=True)

st.divider()

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if st.button("Predict Conversion", type="primary"):
    payload = {
        "age": age,
        "current_occupation": current_occupation,
        "first_interaction": first_interaction,
        "profile_completed": profile_completed,
        "website_visits": website_visits,
        "time_spent_on_website": time_spent_on_website,
        "page_views_per_visit": page_views_per_visit,
        "last_activity": last_activity,
        "print_media_type1": print_media_type1,
        "print_media_type2": print_media_type2,
        "digital_media": digital_media,
        "educational_channels": educational_channels,
        "referral": referral,
    }

    try:
        response = requests.post(BACKEND_URL, json=payload, timeout=15)

        if response.status_code == 200:
            try:
                result = response.json()
                prediction = result["prediction"]
                prediction_label = result["prediction_label"]
                conversion_probability = result["conversion_probability"]
            except (ValueError, KeyError):
                st.error(
                    "The backend returned an unexpected response format. "
                    "Please verify the Flask API is running the correct version."
                )
            else:
                st.subheader("Prediction Result")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric("Conversion Probability", f"{conversion_probability:.1%}")
                with res_col2:
                    if prediction == 1:
                        st.success(f"Prediction: {prediction_label}")
                    else:
                        st.info(f"Prediction: {prediction_label}")

                if prediction == 1:
                    st.write(
                        "This lead shows characteristics associated with a **higher** "
                        "likelihood of conversion based on the model's training data. "
                        "Consider prioritizing this lead for follow-up."
                    )
                else:
                    st.write(
                        "This lead shows characteristics associated with a **lower** "
                        "likelihood of conversion based on the model's training data. "
                        "It may still convert with the right outreach, but may warrant "
                        "less immediate priority than higher-probability leads."
                    )

                st.caption(
                    "This is a machine-learning estimate based on historical patterns, "
                    "not a guarantee of conversion. It is meant to support, not replace, "
                    "the sales team's own judgment."
                )

        elif response.status_code == 400:
            try:
                error_detail = response.json().get("error", "Invalid input.")
            except ValueError:
                error_detail = "Invalid input."
            st.warning(f"The backend rejected this input: {error_detail}")

        elif response.status_code == 500:
            st.error(
                "The backend encountered an internal error while generating the "
                "prediction. Please try again, or contact the team if this persists."
            )

        else:
            st.error(f"Unexpected response from backend (HTTP {response.status_code}).")

    except requests.exceptions.ConnectTimeout:
        st.error("Connection to the backend timed out. Is the Flask API running?")
    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not connect to the backend at {BACKEND_URL}. "
            "Please verify the Flask API is running and BACKEND_URL is set correctly."
        )
    except requests.exceptions.Timeout:
        st.error("The backend took too long to respond. Please try again.")
    except requests.exceptions.RequestException:
        st.error("An unexpected network error occurred while contacting the backend.")

st.divider()
st.caption(
    "ExtraaLearn Lead Conversion Predictor \u2014 powered by a Random Forest model "
    "served through a Flask API. For internal sales-support use."
)
