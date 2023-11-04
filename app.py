import streamlit as st
import pandas as pd
import numpy as np
import joblib
from prediction import predict

st.title('Classifying Dry Bean')
st.header('Bean Features')

col1, col2 = st.columns(2)
with col1:
    area = st.slider('Area', 20300.0, 255000.0, step=10.0)
    perimeter = st.slider('Perimeter', 520.0, 2000.0)
    majorAxisLength = st.slider('MajorAxisLength', 180.0, 740.0)
    minorAxisLength = st.slider('MinorAxisLength', 120.0, 470.0)
    aspectRation = st.slider('AspectRation',1.0, 2.5)
    eccentricity = st.slider('Eccentricity', 0.2, 1.0)
    convexArea = st.slider('ConvexArea', 20600.0, 263300.0)
    equivDiameter = st.slider('EquivDiameter', 160.0, 570.0)
with col2:
    extent = st.slider('Extent', 0.5, 1.0)
    solidity = st.slider('Solidity', 0.9, 1.0)
    roundness = st.slider('Roundness', 0.4, 1.0)
    compactness = st.slider('Compactness', 0.6, 1.0)
    shapeFactor1_imput = st.text_input('ShapeFactor1', placeholder='[0.002, 0.011]')
    shapeFactor2_input = st.text_input('ShapeFactor2', placeholder='[0.0005, 0.04]')
    shapeFactor3_input = st.text_input('ShapeFactor3',placeholder='[0.4, 0.98]')
    shapeFactor4_input = st.text_input('ShapeFactor4', placeholder='[0.94, 1]')
    try:
        shapeFactor1 = float(shapeFactor1_imput)
        shapeFactor2 = float(shapeFactor2_input)
        shapeFactor3 = float(shapeFactor3_input)
        shapeFactor4 = float(shapeFactor4_input)
    except ValueError:
        st.write("Please enter a valid number for these input above")
st.text('')

if st.button("Predict type of Dry Bean"):
    result = predict(np.array([[area, perimeter, majorAxisLength, minorAxisLength, aspectRation, eccentricity, convexArea, equivDiameter, extent, solidity, roundness, compactness, shapeFactor1, shapeFactor2, shapeFactor3, shapeFactor4]]))
    if result == 0:
        st.title('BARBUNYA', )
    if result == 1:
        st.title('BOMBAY')
    if result == 2:
        st.title('CALI')
    if result == 3:
        st.title('DERMASON')
    if result == 4:
        st.title('HOROZ')
    if result == 5:
        st.title('SEKER')
    if result == 6:
        st.title('SIRA')