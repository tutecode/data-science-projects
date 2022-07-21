import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import read_excel
import os
from pytz import country_names
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
from pandas import read_excel
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.



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
       'DP_DISCHARGETIME', 'DP_DISCHARGEQUERIES', 'DP_DISCHARGEPROCESS', 'NPS']

data = read_excel('data-prediction-strata-hospitals.xlsx',
                   header=None,
                   names=column_names)

# Show data head
#print(data.head(10))
#print(data.columns)

# Dimension of the dataset
#print(np.shape(data))

# Let's summarize the data to see the distribution of data
#print(data.describe())


#st.write("""
## Strata Hospitals NPS Prediction App
#This app predicts the **Net Promoter Score**!
#""")
#st.write('---')

fig, axs = plt.subplots(ncols=7, nrows=6, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k, v in data.items():
    sns.boxplot(y=k, data=data, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)