import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
import os
path = kagglehub.dataset_download("imoore/age-dataset")

df = pd.read_csv(os.path.join(path, "AgeDataset-V1.csv"))


from sklearn.model_selection import train_test_split

X = df[["Birth year"]]

y = df[["Age of death"]]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)


st.title("Death predict")

input = st.number_input("Enter your birth year: ")


if st.button("Predict"):

    prediction = model.predict([[input]])
    output = round(prediction.item())



    st.subheader("Predicted age of death:")
    st.write(output)

