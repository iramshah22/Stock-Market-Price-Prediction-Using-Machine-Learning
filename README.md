# Stock-Market-Price-Prediction-Using-Machine-Learning

**Overview**
This project involves predicting stock market prices using two different approaches:
- Linear Regression on Tesla stock price data
- Long Short-Term Memory (LSTM) Networks on Google stock price data

Stock market price prediction is a significant challenge in financial forecasting, and this project demonstrates the use of both traditional regression techniques and advanced deep learning methods to tackle the task.

**Dataset**
-- Tesla Stock Price Dataset: This dataset contains historical stock prices of Tesla, including columns such as Date, Open, High, Low, Close, and Adjusted Close.
-- Google Stock Price Dataset: This dataset contains Google's historical stock prices, used for time series forecasting using LSTM models.


**Technologies and Libraries Used**
-> Python
-> Pandas, NumPy for data manipulation and analysis
-> Matplotlib, Plotly for data visualization
-> Scikit-learn for model training and evaluation
-> Keras for building LSTM models


**Process**

1. **Tesla Stock Price Prediction using Linear Regression**
   
Data Preprocessing:
Read and explore Tesla’s stock price dataset.
Converted the Date column to datetime format for proper time series analysis.
Visualized key statistics of stock prices using box plots.

Data Visualization:
Created an interactive time series plot using Plotly to visualize the stock prices over time.

Modeling with Linear Regression:
Split the data into training and test sets.
Applied feature scaling using StandardScaler.
Built a Linear Regression model using Scikit-learn and plotted the predicted vs. actual stock prices for the training dataset.
Evaluated the model using metrics such as R² Score and Mean Squared Error (MSE).

3. **Google Stock Price Prediction using LSTM**

Data Preprocessing:
Loaded Google stock price data and cleaned it by converting the Close price to numeric.
Used MinMaxScaler to normalize the data for better LSTM performance.
Created time steps by transforming the data into a supervised learning problem (X_train and y_train).

Building the LSTM Model:
Built an LSTM model using Keras with multiple LSTM layers and dropout regularization to prevent overfitting.
The model was trained on the Google stock price dataset over 20 epochs with a batch size of 32.

Evaluation:
Used the trained LSTM model to predict stock prices for the test set.
Visualized the model's performance by plotting the Actual vs. Predicted stock prices.

**Results**

Linear Regression: The model was able to capture the general trend of Tesla's stock prices, but since stock prices can be volatile, the linear model may not perform well for longer-term predictions.
LSTM: The LSTM model was more suited for time series data, making it effective at learning the sequential patterns in Google's stock prices and providing more accurate predictions.


**Conclusion**
This project demonstrates two different approaches to stock price prediction. While traditional machine learning methods like Linear Regression are straightforward and interpretable, more complex models like LSTM are better suited for sequential data and can capture time-dependent patterns more effectively. Future improvements could include exploring other advanced models like GRU or integrating additional market indicator.
