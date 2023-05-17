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

<div align="center">
Welcome to the captivating world of 'FastRide: NYC Taxi Trip Duration Prediction'! 
  
Step into the realm of New York City's bustling streets as we unravel the mysteries behind taxi rides. 
</div>

The purpose of the NYC Taxi Trip Duration Prediction work is to develop a machine learning model that accurately predicts the duration of taxi trips in New York City. This work can help businesses operating in the transportation industry to better manage their resources, optimize their operations, and improve customer satisfaction.

By accurately predicting the duration of taxi trips, businesses can better schedule their vehicles and drivers, ensuring that they are dispatched to the right locations at the right times. This can help to minimize wait times for passengers and reduce the likelihood of missed appointments or deadlines.

Furthermore, the work can enable businesses to provide more accurate fare estimates to their customers, which can help to improve customer satisfaction and loyalty. Customers who receive reliable and transparent fare estimates are more likely to trust and continue to use a particular transportation service.

The analysis of a large dataset of NYC taxi trips, which includes information such as pick-up and drop-off locations, pickup time, and other relevant factors, is utilized to develop a machine learning model. The resulting model is then used to predict the duration of taxi trips for new data points. The entire system employs a serverless architecture.

Overall, the work aims to provide a valuable tool for businesses in the transportation industry to improve their operations, enhance customer satisfaction, and ultimately increase their profitability.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- PROJECT SUMMARY -->
## Summary

<p align="center">
  <img src="https://github.com/kaustubhbhavsar/nyc-taxi-trip-duration/blob/main/assets/architecture.svg">
</p>

<p align="center">
  fig. FastRide Serverless Architecture
</p>

The dataset utilized is sourced from the NYC Taxi Trip Duration competition on Kaggle. It focuses exclusively on the training dataset throughout the project. This training dataset consists of 1,458,644 records, encompassing data from January to June of 2016. The Original dataset is not provided here due to its large size. It can be downloaded from <a href="https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data">here</a>.

Model evaluation is performed using Cross-Validation, incorporating TimeSeriesSplits (note: the kaggle competition didn't take into account splitting the data at the point in time). The Cross-Validation process utilizes data from January to May for training and validating the model, while the data from June is exclusively reserved for testing the final model. For more information on dataset, please refer <a href="datasetinfo.txt">datasetinfo.txt</a>.

To gain a deeper understanding of the dataset, a comprehensive analysis has been conducted and is presented within the <a href="notebooks/data_analysis_and_processing.ipynb">data_analysis_and_processing.ipynb</a> file. The notebook provides a thorough exploration of our dataset, presenting detailed visualizations, statistical analysis, and insightful observations. Within the file, dataset is examined for features, identifying correlations, trends, and potential outliers. 

The <a href="src/features/feature_pipeline.py">feature_pipeline.py</a> script is utilized in the data processing and engineering pipeline for the taxi ride data. It ensures that the relevant features are extracted, combined into an engineered feature matrix, and ultimately stored within the Hopsworks Feature Store, enabling efficient data management and subsequent analysis.

The <a href="src/training/training_part_1.ipynb">training_part_1.ipynb</a> notebook establishes a performance baseline that serves as a benchmark to evaluate the performance of more advanced models such as Linear Regression and HistGradient Boosting. Additionally, the notebook conducts a qualitative analysis to gain deeper insights into the models' performance on the dataset.

Furthermore, <a href="src/training/training_part_2_(pipeline).ipynb">training_part_2_(pipeline).ipynb</a> notebook aims to utilize the preprocessing steps outlined in the preceding notebook in order to build a model using XGBoost. It also incorporates the utilization of weights and biases sweeps. Through this process, the best-performing run's parameters are identified and used to train the final model.

The two Streamlit applications included are (links to public apps given):

*  <a href="src/inference/app.py">app.py</a>: Map-based selection, prediction results, and feature store integration. [<a href="https://kaustubhbhavsar-nyc-taxi-trip-duration-srcinferenceapp-apx1ch.streamlit.app/">link to streamlit prediction app</a>]
*  <a href="src/monitoring/monitoring.py">monitoring.py</a>: Real-time drift calculation and insights (utilizes Alibi Detect framework). [<a href="https://kaustubhbhavsar-nyc-taxi-trip-du-srcmonitoringmonitoring-u0onkv.streamlit.app/">link to streamlit monitoring app</a>]

You can view two recorded screen GIFs demonstrating the functionality of the Streamlit apps below. The GIF on the left demonstrates the prediction app, highlighting the map-based selection, prediction results, and data storage to the feature store. Other GIF on the right showcases the user interface and features of the monitoring app, including real-time drift calculation and insights. Users can monitor and analyze the performance of the prediction model and make informed decisions based on the displayed information.

PREDICTION             |  MONITORING
:-------------------------:|:-------------------------:
![Prediction Demo](https://github.com/kaustubhbhavsar/nyc-taxi-trip-duration/blob/main/assets/webapp_prediction.gif) | ![Monitoring Demo](https://github.com/kaustubhbhavsar/nyc-taxi-trip-duration/blob/main/assets/webapp_monitoring.gif)

### Code and Data Testing

*  The project incorporates <a href="tests/great_expectations/">data testing</a> using the Great Expectations library. 
*  The <a href="tests/code/src/">code testing</a> phase of the project involves the utilization of the Pytest framework.

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

Ensure that all the necessary dependencies and libraries are installed. Refer to the <a href="requirements.txt">requirements.txt</a> file for a complete list of required libraries and their versions. The codebase relies on Python version 3.7.16.

Please note that certain files within this project necessitate the usage of Hopsworks that requires an API key for access. Additionally, an API key for Weights & Biases is required. To proceed with experimentation on a local setup, kindly create an API key and provide it either as a string or through a file.

As noted above, this project includes two Streamlit applications (<a href="src/inference/app.py">app.py</a> and <a href="src/monitoring/monitoring.py">monitoring.py</a>), that may experience latency, typically up to a minute or two, particularly when accessing datasets from Hopsworks. To optimize the performance consider sampling a subset of the data or aggregating it to a manageable size while still maintaining its representative nature.

While random generation of production data (<a href="src/inference/automated_data_generation.py">automated_data_generation.py</a>) is carried out according to a predetermined schedule, it is important to note that this approach does not accurately simulate real-life traffic patterns. Therefore, it should not be employed as a benchmark for comparing against real-life models.

The codebase has been meticulously documented, incorporating comprehensive docstrings and comments. Please review these annotations, as they provide valuable insights into the functionality and operation of the code. 

### Learnings and Limitations

*   Presently, the entirety of the project relies on a single requirements file. Nevertheless, it is highly recommended to employ separate requirement lists for various sections of the project. This approach is particularly beneficial due to the impracticality of employing the installation of same set of libraries for all the tasks incorporated throughout the project. As an illustration, when automating the task of generating production data using Github Action, certain library installations are necessary to run the script, while others are not. Consequently, it is strongly advised to adopt distinct requirement lists that are meticulously tailored to the specific necessities of each project component.
*   The training process is currently performed on Google Colaboratory notebooks due to the unavailability of sufficient high RAM system for model training. While the choice of training environment may or may not have a significant impact, it is still advisable to implement scripts that ensure the portability and reproducibility of the code. These scripts would enable seamless execution and training across different environments, facilitating easy migration to alternative platforms or systems with higher RAM capabilities in the future.
*   The current monitoring component employs a naive approach (although, a good starting point), which falls short of fulfilling the requirement for an automated task that triggers notifications when, for instance, drift is detected. It is advisable to conduct experiments with different window sizes to enhance its effectiveness. Instead of relying solely on Alibi-Detect, It’s recommended to explore alternative platforms such as WhyLabs.ai, Arize.ai, Superwise.ai, Fiddler, DeepChecks, etc. These platforms offer more comprehensive and advanced capabilities for monitoring and detecting anomalies, making them suitable alternatives for improving the monitoring component. 
*   During the inference phase, when logging the production data back to the feature store, latency is encountered. To mitigate this issue, it is recommended to execute this process in the background, ensuring that it does not impact the user experience. By running the logging process asynchronously or in parallel, the latency can be minimized, allowing for smoother user interactions. Alternatively, exploring alternative approaches to reduce latency is also advisable. This could involve optimizing the data logging pipeline, leveraging caching mechanisms, or considering different data storage solutions that offer lower latency. By addressing the latency concerns, the system can operate more efficiently and provide a seamless user experience.

<p align="right">(<a href="#top">back to top</a>)</p>
