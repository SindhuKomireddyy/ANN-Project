import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model


# Load ANN model
model = load_model("ann_model.keras")


# Load scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# Load selected features after feature engineering
with open("features.pkl", "rb") as file:
    features = pickle.load(file)



st.title("Breast Cancer Prediction using ANN")

st.write("Enter the feature values")


input_data = []


# only selected 20 features will appear
for feature in features:

    value = st.number_input(
        feature,
        value=0.0
    )

    input_data.append(value)



if st.button("Predict"):

    input_array = np.array(
        input_data
    ).reshape(1, -1)


    # scaling
    input_scaled = scaler.transform(
        input_array
    )


    prediction = model.predict(
        input_scaled
    )


    if prediction[0][0] > 0.5:

        st.success(
            "Prediction: Benign Tumor"
        )

    else:

        st.error(
            "Prediction: Malignant Tumor"
        )