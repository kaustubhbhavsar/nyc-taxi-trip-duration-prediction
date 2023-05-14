import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import json
import time
import folium
import streamlit as st
from streamlit_folium import st_folium
from src.utils import app_utils, hopsworks_utils


#This script is a Streamlit application that allows users to predict the duration of a taxi trip in 
#New York City. The user can input the vendor ID, the number of passengers, and the pickup and 
#dropoff locations using a map. The application then uses the input to predict the duration of the 
#taxi trip. The application performs some feature engineering steps before making the prediction.


# function to make header fancy across the app (center aligned)
def print_fancy_header_center(text, font_size=24, color="#ff5f27"):
    res = f'<div style="text-align: center;"><span style="color:{color}; font-size: {font_size}px;">{text}</span></div>'
    st.markdown(res, unsafe_allow_html=True)


# display title
st.title('🚖NYC Taxi Trip Duration🚖')
st.markdown("<h1 style='text-align: center; color: black;'>🚖PREDICTION🚖</h1>", unsafe_allow_html=True)
st.write(36 * "-")


# get inputs from user
with st.form(key="user_inputs"):
    # get vendor-id 
    print_fancy_header_center(" 1️⃣ Enter the Vendor-ID:")
    vendor_id = st.selectbox(
        label='',
        options=(1, 2)
    )
    st.write(36 * "-")

    # get number of passengers 
    print_fancy_header_center("2️⃣ Enter the Number of Passengers:")
    passenger_count = st.selectbox(
        label='',
        options=(1, 2, 3, 4, 5, 6)
    )
    st.write(36 * "-")

    # get pickup and dropoff locations
    print_fancy_header_center('3️⃣ Enter the Pickup and Dropoff Coordinates using the Map:')
    st.write("Wait for the map to load then follow the steps given below:\n1. Click on the desired pickup point to select.\n2. Click the 'Submit' button.\n3. Repeat these steps again to select the destination point.")
    st.write("(By default last predicted coordinates will be loaded. Green coloured values are latest updates.)")
    
    my_map = folium.Map(location=[41, -73.5], zoom_start=8)
    my_map.add_child(folium.LatLngPopup())
    folium.TileLayer('Stamen Terrain').add_to(my_map)
    folium.TileLayer('Stamen Toner').add_to(my_map)
    folium.TileLayer('Stamen Water Color').add_to(my_map)
    folium.TileLayer('cartodbpositron').add_to(my_map)
    folium.TileLayer('cartodbdark_matter').add_to(my_map)
    folium.LayerControl().add_to(my_map)
    coordinates = json.load(open("temp_coordinates.json"))
    res_map = st_folium(my_map, height=300, width=600)

    try:
        new_lat, new_long = res_map["last_clicked"]["lat"], res_map["last_clicked"]["lng"]
        # rewrite lat and long for the older coordinate
        if coordinates["c1"]["time_clicked"] > coordinates["c2"]["time_clicked"]:
            target = "c2"
        else:
            target = "c1"
        coordinates[target] = {
            "lat": new_lat,
            "long": new_long,
            "time_clicked": time.time()
        }
        pickup_latitude, pickup_longitude = coordinates["c1"]["lat"], coordinates["c1"]["long"]
        dropoff_latitude, dropoff_longitude = coordinates["c2"]["lat"], coordinates["c2"]["long"]
        # display selected points
        col1, col2 = st.columns(2)
        with col1:
            st.write("📍 Pickup Coordinates:")
            if target == "c1":
                print_fancy_header_center(text=f"Latitude: {pickup_latitude}", font_size=18, color="#52fa23")
                print_fancy_header_center(text=f"Longitude: {pickup_longitude}", font_size=18, color="#52fa23")
            else:
                st.write(f"Latitude: {pickup_latitude}")
                st.write(f"Longitude: {pickup_longitude}")
        with col2:
            st.write("📍 Destination Coordinates:")
            if target == "c2":
                print_fancy_header_center(text=f"Latitude: {dropoff_latitude}", font_size=18, color="#52fa23")
                print_fancy_header_center(text=f"Longitude: {dropoff_longitude}", font_size=18, color="#52fa23")
            else:
                st.write(f"Latitude: {dropoff_latitude}")
                st.write(f"Longitude: {dropoff_longitude}")
        json.dump(coordinates, open("temp_coordinates.json", "w" )) 
    except Exception as err:
        print(err)
        pass

    submit_button = st.form_submit_button(label='Update Coordinates', use_container_width=True)
    
    # get pickup time 
    print_fancy_header_center("4️⃣ Pickup DateTime (in UTC):")
    pickup_datetime=coordinates["c1"]["time_clicked"]
    st.write(pickup_datetime, "(datetime is updated whenever pickup coordinates are updated)")

    if submit_button:
        df_predictor = app_utils.process_input(
            vendor_id,
            passenger_count,
            pickup_latitude, 
            pickup_longitude, 
            dropoff_latitude, 
            dropoff_longitude,
            pickup_datetime
        )
        df_predictor.to_csv(r'E:\NYC Taxi Trip Duration Prediction\src\inference\selected_features.txt', sep='\t', index=False)


try:  
    # feature engineering steps
    print_fancy_header_center('\n🔧 Feature Engineering')
    # load DataFrame from text file
    df_loaded = pd.read_csv(r'E:\NYC Taxi Trip Duration Prediction\src\inference\selected_features.txt', sep='\t')
    st.dataframe(df_loaded) # print all obtained features df
    
    # prediction steps
    print_fancy_header_center('\n🕚 Trip Duration Prediction')
    if st.button('📡 PRESS TO PREDICT', use_container_width=True):
        st.write("<p style='text-align: center;'>(wait a little)</p>", unsafe_allow_html=True)

        # load DataFrame from text file
        df_loaded = pd.read_csv(r'E:\NYC Taxi Trip Duration Prediction\src\inference\selected_features.txt', sep='\t')
        
        # login to hopsworks pass the project as arguement
        project = hopsworks_utils.login_to_hopsworks(project="nyc_taxi_trip_duration")
    
        # get model
        st.write("<p style='text-align: center;'>(getting model)</p>", unsafe_allow_html=True)
        model = app_utils.get_model(project=project, model_name="final_xgboost", version=1)
        st.write("<p style='text-align: center;'>(model received)</p>", unsafe_allow_html=True)
        
        # make predictions
        prediction = model.predict(df_loaded) # in seconds
        minutes = prediction // 60 # convert seconds to minutes
        remaining_seconds = prediction % 60 # remaining seconds
        st.markdown("<h3 style='text-align: center;'>Prediction: {}minutes{}seconds</h3>".format(minutes, remaining_seconds), unsafe_allow_html=True)
        df_prediction = pd.DataFrame.from_dict({ # create DataFrame from a prediction
            'prediction': prediction
        })

        #update the data to hopsworks feature store
        print_fancy_header_center('\n📡 Saving this Data to a Feature Store')

        # connect to feature store
        st.write("<p style='text-align: center;'>(connecting to hopsworks)</p>", unsafe_allow_html=True)
        try:
            feature_store = project.get_feature_store() 
        except Exception as e:
            raise Exception(f"Error connecting to Feature Store at Hopsworks {project}: {e}")

        # get feature group
        st.write("<p style='text-align: center;'>(inserting to feature store)</p>", unsafe_allow_html=True)
        feature_group_name = "predictions"
        prediction_feature_group = feature_store.get_or_create_feature_group(
            name=feature_group_name, 
            version=1,
            primary_key=['pickup_datetime']
        )
       
        # create DataFrame from a pickup_datetime
        df_pickup_datetime = pd.DataFrame.from_dict({
            'pickup_datetime': [pd.to_datetime(pickup_datetime, unit='s').strftime('%Y-%m-%d %H:%M:%S')]
        })
          
        # concat dataframes
        df_updated = pd.concat([df_pickup_datetime, df_loaded, df_prediction], axis=1)
        
        # insert to feature group
        st.write("<p style='text-align: center;'>(wait a little more)</p>", unsafe_allow_html=True)
        prediction_feature_group.insert(df_updated)
        st.balloons()
        st.write("<p style='text-align: center;'>(successfully inserted to feature store)</p>", unsafe_allow_html=True)
        st.write("<p style='text-align: center;'>(refresh the app to start again)</p>", unsafe_allow_html=True)
except Exception as err:
    print(err)
    pass