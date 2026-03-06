#Streamlit UI
#import necessary libraries
import pickle
import streamlit as st 
import pandas as pd
import numpy as np


# model de-serialization also called as loading module
with open("Linear_model.pkl","rb") as file:
    model = pickle.load(file)

#model.predict(data)

# import joblib
# file="model.pkl"
# model=joblib.load(file)
# #model.predict(data)

#encoder de-serializationn(loading encoder)
with open("label_encoder.pkl","rb") as file1:
    encoder = pickle.load(file1)

df=pd.read_csv("cleaned_data.csv")
st.set_page_config(page_title="House price prediction in Banglore",
                   page_icon="houselogo.jpg")
with st.sidebar:
    st.title("Banglore House Price Prediction")
    st.image("https://static.vecteezy.com/system/resources/thumbnails/008/075/444/small/the-logo-of-home-housing-residents-real-estate-with-a-concept-that-presents-rural-nature-with-a-touch-of-leaves-and-sunflowers-vector.jpg")


#input fields
# trained seq : 'bhk','total_sqft','bath','encoded_loc'
location= st.selectbox("Location",options=df["location"].unique())

bhk= st.selectbox("BHK",options=sorted(df["bhk"].unique()))

sqft= st.number_input("Total Sqft: ",min_value=300)

bath= st.selectbox("No. of Rest Rooms ",options=sorted(df["bath"].unique()))
#  ENCODE THE NEW LOACTION
encoded_loc = encoder.transform([location])
# st.write(encoded_loc)

# NEW DATA PREPARATION
new_data = [[bhk,sqft,bath,encoded_loc[0]]]

col1,col2=st.columns([1,2])
#PREDICTION
if col2.button("Predict House Price"):
    pred =model.predict(new_data)[0]
    pred = round(pred*100000)
    st.subheader(f"Predicted Price is : Rs .{pred}")
