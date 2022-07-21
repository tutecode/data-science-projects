import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
from pandas import read_excel
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Strata Hospitals NPS Prediction App
This app predicts the **Net Promoter Score**!
""")
st.write('---')


# Lets load the dataset and sample some
column_names = ['CE_ACCESSIBILITY', 'CE_CSAT', 'CE_VALUEFORMONEY',
       'EM_IMMEDIATEATTENTION', 'EM_NURSING', 'EM_DOCTOR', 'EM_OVERALL',
       'AD_TIME', 'AD_TARRIFFPACKAGESEXPLAINATION', 'AD_STAFFATTITUDE',
       'INR_ROOMCLEANLINESS', 'INR_ROOMPEACE', 'INR_ROOMEQUIPMENT',
       'INR_ROOMAMBIENCE', 'FNB_FOODQUALITY', 'FNB_FOODDELIVERYTIME',
       'FNB_DIETICIAN', 'FNB_STAFFATTITUDE', 'AE_ATTENDEECARE',
       'AE_PATIENTSTATUSINFO', 'AE_ATTENDEEFOOD', 'DOC_TREATMENTEXPLAINATION',
       'DOC_ATTITUDE', 'DOC_VISITS', 'DOC_TREATMENTEFFECTIVENESS',
       'NS_CALLBELLRESPONSE', 'NS_NURSESATTITUDE', 'NS_NURSEPROACTIVENESS',
       'NS_NURSEPATIENCE', 'OVS_OVERALLSTAFFATTITUDE',
       'OVS_OVERALLSTAFFPROMPTNESS', 'OVS_SECURITYATTITUDE',
       'DP_DISCHARGETIME', 'DP_DISCHARGEQUERIES', 'DP_DISCHARGEPROCESS']

# Loads the Strata-Hospitals Dataset
nps = read_excel('data-prediction-strata-hospitals.xlsx',
                   header=None,
                   names=column_names)


X = pd.DataFrame(nps.data, columns=nps.feature_names)
Y = pd.DataFrame(nps.target, columns=["NPS"])

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')


def user_input_features():
    '''
        Return:
            (min, max, media)
                     , default
    '''

    CE_ACCESSIBILITY = st.sidebar.slider('CE_ACCESSIBILITY', X.CE_ACCESSIBILITY.min(), X.CE_ACCESSIBILITY.max(), X.CE_ACCESSIBILITY.mean())
    CE_CSAT = st.sidebar.slider('CE_CSAT', X.CE_CSAT.min(), X.CE_CSAT.max(), X.CE_CSAT.mean())
    CE_VALUEFORMONEY = st.sidebar.slider('CE_VALUEFORMONEY', X.CE_VALUEFORMONEY.min(), X.CE_VALUEFORMONEY.max(),
                              X.CE_VALUEFORMONEY.mean())
    EM_IMMEDIATEATTENTION = st.sidebar.slider('EM_IMMEDIATEATTENTION', X.EM_IMMEDIATEATTENTION.min(), X.EM_IMMEDIATEATTENTION.max(), X.EM_IMMEDIATEATTENTION.mean())
    EM_NURSING = st.sidebar.slider('EM_NURSING', X.EM_NURSING.min(), X.EM_NURSING.max(), X.EM_NURSING.mean())
    EM_DOCTOR = st.sidebar.slider('EM_DOCTOR', X.EM_DOCTOR.min(), X.EM_DOCTOR.max(), X.EM_DOCTOR.mean())
    EM_OVERALL = st.sidebar.slider('EM_OVERALL', X.EM_OVERALL.min(), X.EM_OVERALL.max(), X.EM_OVERALL.mean())
    AD_TIME = st.sidebar.slider('AD_TIME', X.AD_TIME.min(), X.AD_TIME.max(), X.AD_TIME.mean())
    AD_TARRIFFPACKAGESEXPLAINATION = st.sidebar.slider('AD_TARRIFFPACKAGESEXPLAINATION', X.AD_TARRIFFPACKAGESEXPLAINATION.min(), X.AD_TARRIFFPACKAGESEXPLAINATION.max(), X.AD_TARRIFFPACKAGESEXPLAINATION.mean())
    AD_STAFFATTITUDE = st.sidebar.slider('AD_STAFFATTITUDE', X.AD_STAFFATTITUDE.min(), X.AD_STAFFATTITUDE.max(), X.AD_STAFFATTITUDE.mean())
    INR_ROOMCLEANLINESS = st.sidebar.slider('INR_ROOMCLEANLINESS', X.INR_ROOMCLEANLINESS.min(), X.INR_ROOMCLEANLINESS.max(),
                                X.INR_ROOMCLEANLINESS.mean())
    INR_ROOMPEACE = st.sidebar.slider('INR_ROOMPEACE', X.INR_ROOMPEACE.min(), X.INR_ROOMPEACE.max(), X.INR_ROOMPEACE.mean())
    INR_ROOMEQUIPMENT = st.sidebar.slider('INR_ROOMEQUIPMENT', X.INR_ROOMEQUIPMENT.min(), X.INR_ROOMEQUIPMENT.max(), X.INR_ROOMEQUIPMENT.mean())
    INR_ROOMAMBIENCE = st.sidebar.slider('INR_ROOMAMBIENCE', X.INR_ROOMAMBIENCE.min(), X.INR_ROOMAMBIENCE.max(), X.INR_ROOMAMBIENCE.mean())
    FNB_FOODQUALITY = st.sidebar.slider('FNB_FOODQUALITY', X.FNB_FOODQUALITY.min(), X.FNB_FOODQUALITY.max(), X.FNB_FOODQUALITY.mean())
    FNB_FOODDELIVERYTIME = st.sidebar.slider('FNB_FOODDELIVERYTIME', X.FNB_FOODDELIVERYTIME.min(), X.FNB_FOODDELIVERYTIME.max(), X.FNB_FOODDELIVERYTIME.mean())
    FNB_DIETICIAN = st.sidebar.slider('FNB_DIETICIAN', X.FNB_DIETICIAN.min(), X.FNB_DIETICIAN.max(), X.FNB_DIETICIAN.mean())                   
    FNB_STAFFATTITUDE = st.sidebar.slider('FNB_STAFFATTITUDE', X.FNB_STAFFATTITUDE.min(), X.FNB_STAFFATTITUDE.max(), X.FNB_STAFFATTITUDE.mean())
    AE_ATTENDEECARE = st.sidebar.slider('AE_ATTENDEECARE', X.AE_ATTENDEECARE.min(), X.AE_ATTENDEECARE.max(), X.AE_ATTENDEECARE.mean())
    AE_PATIENTSTATUSINFO = st.sidebar.slider('AE_PATIENTSTATUSINFO', X.AE_PATIENTSTATUSINFO.min(), X.AE_PATIENTSTATUSINFO.max(), X.AE_PATIENTSTATUSINFO.mean())
    AE_ATTENDEEFOOD = st.sidebar.slider('AE_ATTENDEEFOOD', X.AE_ATTENDEEFOOD.min(), X.AE_ATTENDEEFOOD.max(), X.AE_ATTENDEEFOOD.mean())
    DOC_TREATMENTEXPLAINATION = st.sidebar.slider('DOC_TREATMENTEXPLAINATION', X.DOC_TREATMENTEXPLAINATION.min(), X.DOC_TREATMENTEXPLAINATION.max(), X.DOC_TREATMENTEXPLAINATION.mean())
    DOC_ATTITUDE = st.sidebar.slider('DOC_ATTITUDE', X.DOC_ATTITUDE.min(), X.DOC_ATTITUDE.max(), X.DOC_ATTITUDE.mean())
    DOC_VISITS = st.sidebar.slider('DOC_VISITS', X.DOC_VISITS.min(), X.DOC_VISITS.max(), X.DOC_VISITS.mean())
    DOC_TREATMENTEFFECTIVENESS = st.sidebar.slider('DOC_TREATMENTEFFECTIVENESS', X.DOC_TREATMENTEFFECTIVENESS.min(), X.DOC_TREATMENTEFFECTIVENESS.max(), X.DOC_TREATMENTEFFECTIVENESS.mean())
    NS_CALLBELLRESPONSE = st.sidebar.slider('NS_CALLBELLRESPONSE', X.NS_CALLBELLRESPONSE.min(), X.NS_CALLBELLRESPONSE.max(), X.NS_CALLBELLRESPONSE.mean())
    NS_NURSESATTITUDE = st.sidebar.slider('NS_NURSESATTITUDE', X.NS_NURSESATTITUDE.min(), X.NS_NURSESATTITUDE.max(), X.NS_NURSESATTITUDE.mean())
    NS_NURSEPROACTIVENESS = st.sidebar.slider('NS_NURSEPROACTIVENESS', X.NS_NURSEPROACTIVENESS.min(), X.NS_NURSEPROACTIVENESS.max(), X.NS_NURSEPROACTIVENESS.mean())
    NS_NURSEPATIENCE = st.sidebar.slider('NS_NURSEPATIENCE', X.NS_NURSEPATIENCE.min(), X.NS_NURSEPATIENCE.max(), X.NS_NURSEPATIENCE.mean())
    OVS_OVERALLSTAFFATTITUDE = st.sidebar.slider('OVS_OVERALLSTAFFATTITUDE', X.OVS_OVERALLSTAFFATTITUDE.min(), X.OVS_OVERALLSTAFFATTITUDE.max(), X.OVS_OVERALLSTAFFATTITUDE.mean())
    OVS_OVERALLSTAFFPROMPTNESS = st.sidebar.slider('OVS_OVERALLSTAFFPROMPTNESS', X.OVS_OVERALLSTAFFPROMPTNESS.min(), X.OVS_OVERALLSTAFFPROMPTNESS.max(), X.OVS_OVERALLSTAFFPROMPTNESS.mean())
    OVS_SECURITYATTITUDE = st.sidebar.slider('OVS_SECURITYATTITUDE', X.OVS_SECURITYATTITUDE.min(), X.OVS_SECURITYATTITUDE.max(), X.OVS_SECURITYATTITUDE.mean())
    DP_DISCHARGETIME = st.sidebar.slider('DP_DISCHARGETIME', X.DP_DISCHARGETIME.min(), X.DP_DISCHARGETIME.max(), X.DP_DISCHARGETIME.mean())           
    DP_DISCHARGEQUERIES = st.sidebar.slider('DP_DISCHARGEQUERIES', X.DP_DISCHARGEQUERIES.min(), X.DP_DISCHARGEQUERIES.max(), X.DP_DISCHARGEQUERIES.mean())
    DP_DISCHARGEPROCESS = st.sidebar.slider('DP_DISCHARGEPROCESS', X.DP_DISCHARGEPROCESS.min(), X.DP_DISCHARGEPROCESS.max(), X.DP_DISCHARGEPROCESS.mean())
    data = {'CE_ACCESSIBILITY': CE_ACCESSIBILITY,
            'CE_CSAT': CE_CSAT,
            'CE_VALUEFORMONEY': CE_VALUEFORMONEY,
            'EM_IMMEDIATEATTENTION': EM_IMMEDIATEATTENTION,
            'EM_NURSING': EM_NURSING,
            'EM_DOCTOR': EM_DOCTOR,
            'EM_OVERALL': EM_OVERALL,
            'AD_TIME': AD_TIME,
            'AD_TARRIFFPACKAGESEXPLAINATION': AD_TARRIFFPACKAGESEXPLAINATION,
            'AD_STAFFATTITUDE': AD_STAFFATTITUDE,
            'INR_ROOMCLEANLINESS': INR_ROOMCLEANLINESS,
            'INR_ROOMPEACE': INR_ROOMPEACE,
            'INR_ROOMEQUIPMENT': INR_ROOMEQUIPMENT,
            'INR_ROOMAMBIENCE': INR_ROOMAMBIENCE,
            'FNB_FOODQUALITY': FNB_FOODQUALITY,
            'FNB_FOODDELIVERYTIME': FNB_FOODDELIVERYTIME,
            'FNB_DIETICIAN': FNB_DIETICIAN,
            'FNB_STAFFATTITUDE': FNB_STAFFATTITUDE,
            'AE_ATTENDEECARE': AE_ATTENDEECARE,
            'AE_PATIENTSTATUSINFO': AE_PATIENTSTATUSINFO,
            'AE_ATTENDEEFOOD': AE_ATTENDEEFOOD,
            'DOC_TREATMENTEXPLAINATION': DOC_TREATMENTEXPLAINATION,
            'DOC_ATTITUDE': DOC_ATTITUDE,
            'DOC_VISITS': DOC_VISITS,
            'DOC_TREATMENTEFFECTIVENESS': DOC_TREATMENTEFFECTIVENESS,
            'NS_CALLBELLRESPONSE': NS_CALLBELLRESPONSE,
            'NS_NURSESATTITUDE': NS_NURSESATTITUDE,
            'NS_NURSEPROACTIVENESS': NS_NURSEPROACTIVENESS,
            'NS_NURSEPATIENCE': NS_NURSEPATIENCE,
            'OVS_OVERALLSTAFFATTITUDE': OVS_OVERALLSTAFFATTITUDE,
            'OVS_OVERALLSTAFFPROMPTNESS': OVS_OVERALLSTAFFPROMPTNESS,
            'OVS_SECURITYATTITUDE': OVS_SECURITYATTITUDE,
            'DP_DISCHARGETIME': DP_DISCHARGETIME,
            'DP_DISCHARGEQUERIES': DP_DISCHARGEQUERIES,
            'DP_DISCHARGEPROCESS': DP_DISCHARGEPROCESS}
    features = pd.DataFrame(data, index=[0])

    return features


df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)

# Apply Model to Make Prediction
# Predict: NPS - Net Promoter Score
# Do with Pickle
prediction = model.predict(df)

st.header('Prediction of NPS')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
st.write("""

""")
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')
