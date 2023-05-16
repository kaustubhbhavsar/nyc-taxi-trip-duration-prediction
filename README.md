<!-- PROJECT NAME -->

<br />
<div align="center">
  <h3 align="center">FastRide: NYC Taxi Trip Duration Prediction</h3>
  <p align="center">
      <b>Weights&amp;Biases</b> Experiment Tracking | <b>Hopsworks</b> Feature Store | <b>Alibi-Detect</b> Monitoring | <b>Streamlit</b> Web-App 
  </p>
</div>

<!-- ABOUT PROJECT -->
## What Is It?

The purpose of the NYC Taxi Trip Duration Prediction work is to develop a machine learning model that accurately predicts the duration of taxi trips in New York City. This work can help businesses operating in the transportation industry to better manage their resources, optimize their operations, and improve customer satisfaction.

By accurately predicting the duration of taxi trips, businesses can better schedule their vehicles and drivers, ensuring that they are dispatched to the right locations at the right times. This can help to minimize wait times for passengers and reduce the likelihood of missed appointments or deadlines.

Furthermore, the work can enable businesses to provide more accurate fare estimates to their customers, which can help to improve customer satisfaction and loyalty. Customers who receive reliable and transparent fare estimates are more likely to trust and continue to use a particular transportation service.

The analysis of a large dataset of NYC taxi trips, which includes information such as pick-up and drop-off locations, pickup time, and other relevant factors, is utilized to develop a machine learning model. The resulting model is then used to predict the duration of taxi trips for new data points.

Overall, the work aims to provide a valuable tool for businesses in the transportation industry to improve their operations, enhance customer satisfaction, and ultimately increase their profitability.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- PROJECT SUMMARY -->
## Summary

Welcome to the captivating world of 'FastRide: NYC Taxi Trip Duration Prediction'! Step into the realm of New York City's bustling streets as we unravel the mysteries behind taxi rides. 

The dataset utilized is sourced from the NYC Taxi Trip Duration competition on Kaggle. It focuses exclusively on the training dataset throughout the project. This training dataset consists of 1,458,644 records, encompassing data from January to June of 2016.

Model evaluation is performed using Cross-Validation, incorporating TimeSeriesSplits. The Cross-Validation process utilizes data from January to May for training and validating the model, while the data from June is exclusively reserved for testing the final model. For more information on dataset, please refer <a href="datasetinfo.txt">datasetinfo.txt</a>.

To gain a deeper understanding of the dataset, a comprehensive analysis has been conducted and is presented within the <a href="src/notebooks/data_analysis_and_processing.ipynb'">data_analysis_and_processing.ipynb</a> file. The notebook provides a thorough exploration of our dataset, presenting detailed visualizations, statistical analysis, and insightful observations. Within the file, dataset is examined for features, identifying correlations, trends, and potential outliers. 

The <a href="src/features/feature_pipeline.py'">feature_pipeline.py</a> script plays a crucial role in the data processing and engineering pipeline for the taxi ride data. It ensures that the relevant features are extracted, combined into an engineered feature matrix, and ultimately stored within the Hopsworks Feature Store, enabling efficient data management and subsequent analysis.

This project includes two Streamlit applications:

*  <a href="src/inference/app.py">app.py</a>: Map-based selection, prediction results, and feature store integration.
*  <a href="src/monitoring/monitoring.py">monitoring.py</a>: Real-time drift calculation and insights.

You can view two recorded screen GIFs demonstrating the functionality of the Streamlit apps below. The first GIF demonstrates the prediction app, highlighting the map-based selection, prediction results, and data storage to the feature store. The second GIF showcases the user interface and features of the monitoring app, including real-time drift calculation and insights. Users can monitor and analyze the performance of the prediction model and make informed decisions based on the displayed information.

PREDICTION             |  MONITORING
:-------------------------:|:-------------------------:
![Prediction Demo](https://github.com/kaustubhbhavsar/nyc-taxi-trip-duration/blob/main/assets/webapp_prediction.gif) | ![Monitoring Demo](https://github.com/kaustubhbhavsar/nyc-taxi-trip-duration/blob/main/assets/webapp_monitoring.gif)

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- Project Directory Structure -->
## Directory Structure
```
├── config/                                        # configuration files        
├── model/                                         # locally Saved models              
├── notebooks/                                     # notebooks
    └── data_analysis_and_processing.ipynb         # eda and processing component
    └── ...                                        # other notebooks
├── src/                                           # main code files
    └── features/                                  # features components
    └── inference/                                 # inference components
    └── monitoring/                                # monitoring components
    └── training/                                  # training components
    └── utils/                                     # supplementary utilities
├── tests/                                         # test files
    └── code/                                      # code tests
    └── great_expectations/                        # data tests
```

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- Tools and Libraries used -->
## Tools and Libraries

*   Language: Python
*   Experiment Tracking: Weights and Biases
*   Feature Store: Hopsworks
*   Monitoring: Alibi-Detect
*   Testing:  Great-Expectations, Pytest
*   Web App: Streamlit, Folium
*   Other Prominent Libraries: Scikit-Learn, Xgboost, Pandas, Numpy, Matplotlib, Seaborn, Yellowbrick

The additional libraries utilized, along with the precise versions of each library used, are specified in the <a href="requirements.txt">requirements.txt</a> file.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Final Notes -->
## Final Notes

Ensure that all the necessary dependencies and libraries are installed. Refer to the <a href="requirements.txt">requirements.txt</a> file for a complete list of required libraries and their versions.

Please note that certain files within this project necessitate the usage of Hopsworks. Additionally, an API key for Weights & Biases is required. To proceed with experimentation on a local setup, kindly create an API key and provide it either as a string or through files.

As noted above, this project includes two Streamlit applications (<a href="src/inference/app.py">app.py</a> and <a href="src/monitoring/monitoring.py">monitoring.py</a>), that may experience latency, typically up to a minute or two, particularly when accessing datasets from Hopsworks. To optimize the performance consider sampling a subset of the data or aggregating it to a manageable size while still maintaining its representative nature.

The codebase has been meticulously documented, incorporating comprehensive docstrings and comments. Please review these annotations, as they provide valuable insights into the functionality and operation of the code. 

<p align="right">(<a href="#top">back to top</a>)</p>
