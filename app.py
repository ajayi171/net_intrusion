import streamlit as st
import pickle
import numpy as np
import pandas as pd 
from numpy import hstack
from numpy import vstack
from numpy import asarray
from PIL import Image

st.title("Intrusion Detection App")

att_model = pickle.load(open('att_model.pkl', 'rb'))
intr_model = pickle.load(open('k_model.pkl', 'rb'))
models = pickle.load(open('sl_model.pkl', 'rb'))
meta_model=pickle.load(open('meta_model.pkl', 'rb'))

label_encoder2 = pickle.load(open('le2.pkl', 'rb'))
label_encoder4 = pickle.load(open('le4.pkl', 'rb'))

scaler1 = pickle.load(open('att_scal.pkl', 'rb'))

scaler2 = pickle.load(open('scal.pkl', 'rb'))

encoder = pickle.load(open('enc.pkl', 'rb'))

col3 = ['ct_state_ttl','rate','sttl','dmean','ct_dst_src_ltm',
        'dload','ct_srv_src','sbytes','dur', 'sload', 'tcprtt',
        'ct_srv_dst', 'dbytes', 'smean']

sf = ['dur', 'sbytes', 'dbytes', 'sttl', 'sload', 'dload', 
     'smean', 'dmean', 'ct_srv_src', 'ct_srv_dst']

sf2 =  ['sbytes', 'rate', 'sttl', 'sload', 'dload', 'tcprtt', 
     'smean', 'ct_state_ttl', 'ct_dst_src_ltm', 'ct_srv_dst']


def predict():

    c1,c2,c3 = st.columns(3)

    with c1:
        smean = st.number_input('Smean')
        ct_srv_dst = st.number_input('ct_srv_dst')
        ct_dst_src_ltm = st.number_input('ct_dst_src_ltm')
        sbytes = st.number_input('sbytes')
        dbytes = st.number_input('dbytes')
        
    with c2:
        sttl = st.number_input('sttl')
        dur = st.number_input('dur')
        ct_state_ttl = st.number_input('ct_state_ttl')
        dload = st.number_input('dload')
        dmean = st.number_input('dmean')

    with c3:
        sload = st.number_input('sload')
        tcprtt = st.number_input('tcprtt')
        ct_srv_src = st.number_input('ct_srv_src')
        rate = st.number_input('rate')
        ct_dst_sport_ltm = st.number_input('ct_dst_sport_ltm')

    feat = np.array([ct_state_ttl,rate,sttl,dmean,ct_dst_src_ltm,
                      dload,ct_srv_src,sbytes,dur,sload,tcprtt,
                      ct_srv_dst,dbytes,smean]).reshape(1,-1)

                

    feat1 = pd.DataFrame(feat,columns=col3)
        
    return feat1

def super_learner_predictions(X, models, meta_model):
	meta_X = list()
	for model in models:
		yhat = model.predict_proba(X)
		meta_X.append(yhat)
	meta_X = hstack(meta_X)
	# predict
	return meta_model.predict(meta_X)

data = predict()

def prepare(data):

    attack_df = data[sf]
    #attack_df['service'] = label_encoder2.transform(attack_df['service'])
    intr_df = data[sf2]

    attack_df = scaler1.transform(attack_df)
    intr_df = scaler2.transform(intr_df)

    return attack_df, intr_df

if st.button('Predict'):
    attack_df, intr_df = prepare(data)

    pred = super_learner_predictions(intr_df, models, meta_model)

    if pred[0] == 0:
        st.write("Normal Activity Permision Granted")
    else:
        pred1 = att_model.predict(attack_df)
        attack = label_encoder4.inverse_transform(pred1)
        # st.subheader(f"{attack[0]}")
        st.warning(f"{attack[0]} Intrusion Detected")
        st.image("dz1.gif")

        

