import streamlit as st
import pandas as pd
from f1predictor import f1p


example_data = pd.read_csv('prod_dataset.csv')

st.title('F1 Race Top 5 Position Predictor')
st.write("""
This application predicts the top 5 race positions for the next Formula 1 race based on past performance data.
""")

st.subheader('Past Performance Data')
edited_df = st.data_editor(example_data)

if st.button('Predict positions'):
    predictions = f1p.predict(edited_df)[["driverid", "unique_pred_position"]]
    st.subheader('Predicted Race Positions')
    st.write(predictions)
