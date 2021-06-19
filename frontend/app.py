"""
You need to run the app from the root
To run the app
$ streamlit run serving/frontend/app.py
"""

import numpy as np
import json
import joblib
import pandas as pd
import streamlit as st
from PIL import Image
import requests
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



headers = ['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6', 'quantitative']
url_host = "http://127.0.0.1:8000"

with st.sidebar:
    st.subheader('Instruction!!!!')
    
    st.text("Read the README file!!")
    st.write("[https://github.com/tiennguyenhust/dsp-diabetes-tien](https://github.com/tiennguyenhust/dsp-diabetes-tien)")
    image = Image.open('frontend/images/img.jpg')
    st.image(image, caption='*')
    
    st.subheader('Saved models:')
    st.text("dsp-diabetes-tien/models")

    
st.title('Diabetes Prediction: Sherlock Holmes')
model = None
model_name = ""


st.subheader('Training!')
#  CSV file
train_file = st.file_uploader('Choose a CSV file for data train : Full Data')
if train_file:
    st.write('filename: ', train_file.name)
    data_train = pd.read_csv(train_file, names=headers)
    st.write(data_train)

    st.text("Dimension: " + str(data_train.shape))
    

option_model = st.selectbox('Training model selection: ', ('LinearRegression', 'ElasticNet', 'RandomForestRegressor')) 
alpha = None
l1_ratio = None
if option_model == 'ElasticNet':
    param_cols = st.beta_columns(2)
    with param_cols[0]:
        alpha = st.number_input('alpha', min_value=0.000, max_value=1.000)
    with param_cols[1]:
        l1_ratio = st.number_input('l1_ratio', min_value=0.000, max_value=1.000)  

# training
if st.button('Training'):
    if train_file:
        kwargs = {}
        if alpha and l1_ratio:
            kwargs = {'alpha': alpha, 'l1_ratio': l1_ratio}
            
        data_train=data_train.to_json(orient='records', lines=True).split('\n')
        data_train=[json.loads(i) for i in data_train if i != '']
        data = {'kwargs': kwargs, 'data_train': data_train}
        res = requests.post(url_host + '/training?model_name={}'.format(option_model), json=data).json()
        
        st.success(f'Training Successful! Your model {option_model} is saved! \n' + str(res))
    else:
        st.warning('You need to upload a CSV file')


#==================== Prediction ==============
# List of models available

st.subheader('Choose your model!')

def get_available_models():
    model_names = requests.get(url_host + "/models")
    return model_names.json()


selected_model_name = st.selectbox("Available models: (All saved models are loaded here!)", get_available_models()) 

model_file = st.file_uploader('Upload your own model:')
if model_file:
    if model_file.name[-7:] == '.joblib' or model_file.name[-4:] == '.sav' or model_file.name[-4:] == '.pkl':
        selected_model_name = model_file.name
    else:
        st.warning('Wrong file!!!')
        st.stop()

selected_model = st.markdown('Selected Model: **_{}_**'.format(str(selected_model_name)))

st.subheader('Prediction!')

st.markdown('**_Signle Prediction!_**')

input_expander = st.beta_expander("A Patient Data", expanded=True)
with input_expander:
    col1, col2 = st.beta_columns(2)
    with col1:
        age = st.number_input('age')
        sex = st.number_input('sex')
        bmi = st.number_input('bmi')
        bp = st.number_input('bp')
        s1 = st.number_input('s1')
    with col2:
        s2 = st.number_input('s2')
        s3 = st.number_input('s3')
        s4 = st.number_input('s4')
        s5 = st.number_input('s5')
        s6 = st.number_input('s6')

if st.button('Predict for one patient'):

    if not selected_model_name:
        st.warning("No model existed! Please select your model!")
        st.stop()
    res = requests.post(
        url_host + "/predict?model_name={}&age={}&sex={}&bmi={}&bp={}&s1={}&s2={}&s3={}&s4={}&s5={}&s6={}".format(selected_model_name,age,sex,bmi,bp,s1,s2,s3,s4,s5,s5)
    )
    
    predictions = res.json()
    st.success(f'Predictions : {predictions}')


#  CSV file

st.markdown('**_Multi Prediction!_**')
X_test_file = st.file_uploader('Choose a CSV file for prediction')

col_test, col_pred = st.beta_columns((3, 1))
with col_test:
    if X_test_file:
        st.write('filename: ', X_test_file.name)
        X_test = pd.read_csv(X_test_file, names=headers[0:-1])
        st.write(X_test)
with col_pred:
    if X_test_file:
        st.write('Result')
        st_prediction = st.empty()
    
def inference(selected_model_name):
    data=X_test.to_json(orient='records', lines=True).split('\n')
    data=[json.loads(i) for i in data if i != '']
    
    res = requests.post(url_host + "/predict_obj?model_name={}".format(selected_model_name), json=data)           
    
    return res.json()


if st.button('Predict for multi patients'):
    if not selected_model_name:
        st.warning("No model existed! Please select your model!")
        st.stop()
    if not X_test_file:
        st.warning('Please input the file CSV!')
        st.stop()
        
    res = inference(selected_model_name)

    #predictions = res.split(" ")
    #results = [[i, predictions[i]] for i in range(len(predictions))]
    st.success(f'Successful Prediction!')

    st_prediction.write(pd.DataFrame(res, columns=['Prediction'])) 

"""
To cover
- please make sure that there are NO titles (age, sex, bmi, bp, ... ) in the begining of .csv file
- reading csv one time
- executing only if data is loaded
"""