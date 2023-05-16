import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from alibi_detect.cd import TabularDrift
import pandas as pd
import streamlit as st
from src.utils import hopsworks_utils, monitoring_utils

#This file is a script that performs drift detection on the NYC Taxi Trip Duration prediction model. 
#It uses the streamlit library to display the results in a web application. Specifically, 
#the script calculates the Target Drift and Feature-wise Drift by comparing the distributions of two 
#sets of observations. The two-sample Kolmogorov-Smirnov test is used to detect any drift in these 
#variables.


# function to make header fancy across the app (center aligned)
def print_fancy_header_center(text, font_size=24, color="#ff5f27"):
    res = f'<div style="text-align: center;"><span style="color:{color}; font-size: {font_size}px;">{text}</span></div>'
    st.markdown(res, unsafe_allow_html=True)


# display title
st.markdown("<h1 style='text-align: center; color: black;'>NYC Taxi Trip Duration</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>🚖MONITORING🚖</h1>", unsafe_allow_html=True)
st.write(36 * "-")

st.write("We calculate Target Drift and Feature-wise Drift and use the two-sample Kolmogorov-Smirnov test (at P-value: 0.05) to detect any drift in these variables by comparing the distributions of two sets of observations - reference and test window.", unsafe_allow_html=True)
st.write("Due to the limited number of predictions made so far, the detected drifts may not be entirely reliable, and there may also be discrepancies in the reference window. However, this still provides us with insights into how the system is currently performing (a good starting point).", unsafe_allow_html=True)

try:  
    st.write("<p style='text-align: center;'>(connecting to feature store)</p>", unsafe_allow_html=True)
    try:
       # login to hopsworks pass the project as arguement
        project = hopsworks_utils.login_to_hopsworks(project='nyc_taxi_trip_duration', api_key=st.secrets["HOPSWORKS_API_KEY"])
        # connect to feature store
        fs = project.get_feature_store()
    except Exception as e:
        raise Exception(f"Error connecting to Feature Store in project {project}: {e}")

    # getting data
    st.write("<p style='text-align: center;'>(getting data)</p>", unsafe_allow_html=True)
    st.write("<p style='text-align: center;'>(wait a little)</p>", unsafe_allow_html=True)
    st.write("<p style='text-align: center;'>(this may take a few seconds)</p>", unsafe_allow_html=True)
    test_df = monitoring_utils.get_df(
        feature_store=fs,
        feature_view_name='combined_features',
        feature_view_version=1,
        training_dataset_version=2 # get test dataframe (which is training_dataset_version=2 in feature view)
    )
    st.write("<p style='text-align: center;'>(wait a little more)</p>", unsafe_allow_html=True)
    prediction_df = monitoring_utils.get_df(
        feature_store=fs,
        feature_view_name='prediction',
        feature_view_version=1,
        training_dataset_version=1 
    )

    test_df_selected_columns, prediction_df_selected_columns, selected_features = monitoring_utils.get_specific_features(
        test_df=test_df, 
        prediction_df=prediction_df
    )

    st.write("<p style='text-align: center;'>(calculating drift)</p>", unsafe_allow_html=True)
    # define drift detector with custom categories_per_feature parameter (ref=test_df)
    cd = TabularDrift(test_df_selected_columns, p_val=.05, categories_per_feature={})

    # define dataframes for output
    results_df_pred = pd.DataFrame(columns=['Feature', 'Drift', 'Stat Value', 'P-Value'])
    results_df_rest = pd.DataFrame(columns=['Feature', 'Drift', 'Stat Value', 'P-Value'])

    # predict drift
    fpreds = cd.predict(prediction_df_selected_columns, drift_type='feature')
    labels = ['No', 'Yes'] # 'is drift' label
    for f in range(cd.n_features):
        fname = selected_features[f]
        is_drift = labels[fpreds['data']['is_drift'][f]]
        stat_val = fpreds['data']['distance'][f]
        p_val = fpreds['data']['p_val'][f]
        if fname == 'prediction':
            results_df_pred = results_df_pred.append({'Feature': fname, 'Drift': is_drift, 'Stat Value': stat_val, 'P-Value': p_val}, ignore_index=True)
        else:
            results_df_rest = results_df_rest.append({'Feature': fname, 'Drift': is_drift, 'Stat Value': stat_val, 'P-Value': p_val}, ignore_index=True)

    # target drift
    print_fancy_header_center('\n🔧 Target Drift')
     # format the dataframe to display 20 decimal points
    st.dataframe(results_df_pred, use_container_width=True)

    # other features drift
    print_fancy_header_center('\n🔧 Prominent Features Drift')
    st.dataframe(results_df_rest, use_container_width=True)

    st.write("NOTE: If the p-value is zero, it is not exactly zero but rather a very small value.", unsafe_allow_html=True)

    # show balloons as success
    st.balloons()

except Exception as err:
    print(err)
    pass

