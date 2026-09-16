import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
import os
path = kagglehub.dataset_download("wardabilal/salary-prediction-dataset")

df = pd.read_csv(os.path.join(path, "Salary_Data.csv"))

original_processed_df = df.copy()
df = df.dropna()


from sklearn.model_selection import train_test_split

X = df[["Salary"]]

y = df[["Years of Experience"]]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)


st.title("Experience predict")

input = st.number_input("Enter yearly salary")


if st.button("Predict"):

    prediction = model.predict([[input]])
    output = round(prediction.item())


    st.subheader("Result")

    st.write("Experience years:")
    st.write(output)

