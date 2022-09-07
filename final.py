import streamlit as st
import pandas as pd
from PIL import Image
import pickle
import numpy as np

classifier = pickle.load(open("final_model.pkl", "rb"))

def churn_prediction(input_data):
    array=np.asarray(input_data)
    reshape = array.reshape(1,-1)

    prediction = classifier.predict(reshape)
    print(prediction)

def main():
    st.title("Telecommunications Churn Prediction")

    data = pd.read_csv("cleaned_data.csv")

    if st.button('Show Dataset'):
        st.header('Telecommunication Churn Dataset')
        st.write(data)

    def user_input_features():
        voice_mail_plan = st.sidebar.selectbox("Voice Mail Plan", ("1", "0"))
        voice_mail_messages = st.sidebar.number_input("Number voice mail messages")
        international_mins = st.sidebar.number_input("Total international mins")
        customer_service_calls = st.sidebar.number_input("Number customer service calls")
        international_plan = st.sidebar.selectbox("International Plan", ("1", "0"))
        international_calls = st.sidebar.number_input("Number international calls")
        international_charge = st.sidebar.number_input("Total international charge")
        total_charge = st.sidebar.number_input("Total charge")
        df = {"voice_mail_plan": voice_mail_plan, "voice_mail_messages": voice_mail_messages,
                "international_mins": international_mins,
                "customer_service_calls": customer_service_calls, "international_plan": international_plan,
                "international_calls": international_calls, "international_charge": international_charge,
                "total_charge": total_charge}
        features = pd.DataFrame(df, index=[0])
        return features

    result = user_input_features()

    st.subheader("User Input Parameters")
    st.write(result)

    st.subheader('Predicted Result')
    image1 = Image.open('Thumbs up.jpg')
    image2 = Image.open('Thumbs down.jpg')
    customer_churn = ""

    if st.button('Predict'):
        customer_churn = churn_prediction(df)
        if (prediction[0][1] == 0):
            st.image(image1)
            return "The customer is loyal."
        else:
            st.image(image2)
            return "The customer is churn."

    st.success(customer_churn)

if __name__ == "__main__":
    main()

    