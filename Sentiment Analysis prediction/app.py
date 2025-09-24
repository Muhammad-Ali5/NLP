import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model and vectorizer
model = joblib.load("my_Trained_joblib_model.joblib")
vectorizer = joblib.load("vectorizer_joblib.joblib")

# Global list to store results for visualization
predictions_list = []

#  Set up the page
st.set_page_config(page_title="💬 Tweet Sentiment Analyzer", layout="centered")
st.markdown("<h1 style='text-align: center; color: #FF69B4;'>💖 Tweet Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6c757d;'>🔍 Analyze the sentiment of your tweet in one click!</p>", unsafe_allow_html=True)
st.markdown("---")

#  User input
user_input = st.text_input("📝 Type your tweet here:")

# Predict Button
if st.button(" Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a tweet first!")
    else:
        # Transform and predict
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)[0]

        # Store prediction for chart
        predictions_list.append(prediction)

        # 💡 Display result
        if prediction == 1:
            st.success(" Positive Tweet Detected! 😊")
            st.balloons()
        else:
            st.error("😢 Negative Tweet Detected.")

#  Show Pie Chart after at least 1 prediction
if len(predictions_list) > 0:
    st.markdown("---")
    st.subheader("📊 Sentiment Prediction Summary")

    pos_count = predictions_list.count(1)
    neg_count = predictions_list.count(0)

    labels = ["Positive 😊", "Negative 😞"]
    sizes = [pos_count, neg_count]
    colors = ['#90ee90', '#ff9999']

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')

    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("<small style='text-align:center; display:block;'></small>", unsafe_allow_html=True)
