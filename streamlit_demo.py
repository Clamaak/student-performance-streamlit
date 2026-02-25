import streamlit as st
import pandas as pd
st.write('Streamlit test!!')

st.write('Streamlit is working fine!')

df = pd.read_csv('student_performance.csv')
st.write(df)