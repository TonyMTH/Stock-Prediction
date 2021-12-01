import datetime

import streamlit as st
import pandas as pd

from predict import PredictStock

# Text/Title
st.title("Stock Prediction ")
st.write('Data must be in csv file and of this order:\n\n"date","ISE","ISE","SP","DAX","FTSE","NIKKEI","BOVESPA",'
         '"EU","EM"\n')

model_pth = 'data/best_model.pt'

uploaded_file = st.file_uploader('Upload Stock File')
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.dataframe(data)

    if st.button('Predict'):
        out = PredictStock(model_pth).predict(data)

        date = datetime.datetime.strptime(data['date'].values[-1], "%d-%b-%y")
        date = date + datetime.timedelta(days=1)

        st.write('ISE.1 Market Price for {} is:\t\t{:.4f}'.format(str(date).split(" ")[0], out))

else:
    st.write(f'Please upload a file')

