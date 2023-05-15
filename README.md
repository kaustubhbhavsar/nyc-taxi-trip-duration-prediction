<!-- PROJECT NAME -->

<br />
<div align="center">
  <h3 align="center">NYC Taxi Trip Duration</h3>
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

The additional libraries utilized, along with the precise versions of each library used, are specified in the requirements.txt file.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Final Notes -->
## Final Notes

Ensure that all the necessary dependencies and libraries are installed. Refer to the "Requirements" section in the documentation for a complete list of required libraries and their versions.

Please note that certain files within this project necessitate the usage of Hopsworks. Additionally, an API key for Weights & Biases is required. To proceed with experimentation on a local setup, kindly create an API key and provide it either as a string or through files.

This project includes two Streamlit applications:

*  <a href="src/inference/app.py">app.py</a>: This application is designed for making predictions.
*  <a href="src/monitoring/monitoring.py">monitoring.py</a>: This application is intended for monitoring purposes.

Both app.py and monitoring.py may experience latency due to various factors. The performance of the applications, particularly when accessing datasets from Hopsworks, can be influenced by the size of the datasets. To optimize the performance consider sampling a subset of the data or aggregating it to a manageable size while still maintaining its representative nature.

The codebase has been meticulously documented, incorporating comprehensive docstrings and comments. Please review these annotations, as they provide valuable insights into the functionality and operation of the code. 

<p align="right">(<a href="#top">back to top</a>)</p>
