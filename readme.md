Flight Price Forecasting Using LSTM
===================================

This project predicts future flight ticket prices using a multivariate, multi-step LSTM model built with TensorFlow/Keras. The model is trained on historical ticket data and can forecast future prices based on route, calendar features, and more.

Project Structure
-----------------
- ticket_price_dist.csv       <- Raw dataset containing historical ticket pricing
- main.ipynb                  <- Jupyter Notebook for data preprocessing, training, and evaluation
- model/                      <- (Optional) Directory to store saved models or logs
- README.txt                  <- This file

Goals
-----
- Predict the next 7 days of flight ticket prices (`FORECAST_HORIZON = 7`)
- Use multiple features (route, flight time, calendar, etc.)
- Evaluate prediction accuracy using MAE, RMSE, MSE
- Plot results and monitor overfitting with validation loss

Preprocessing
-------------
- Columns used: 
    - calendar_year, calendar_month, week_number
    - weekday_name, flight_time_hour, distance_km
    - route (origin + destination), days_until_departure
- Normalization: 
    - Numerical features scaled with MinMaxScaler
    - Categorical features one-hot encoded
- Time-series Sequences:
    - WINDOW_SIZE: 14 (past 14 days as input)
    - FORECAST_HORIZON: 7 (predict 7 days ahead)
    - Grouped by route to preserve temporal structure

Model Architecture
------------------
- LSTM(64)
- Dropout(0.3)
- Dense(64, activation='relu')
- Dropout(0.3)
- Dense(FORECAST_HORIZON)

Training Setup
--------------
- Optimizer: Adam
- Loss: Mean Squared Error (MSE)
- Callbacks:
    - EarlyStopping(patience=10, restore_best_weights=True)
    - ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

Evaluation
----------
- Metrics: MAE, RMSE, MAPE
- Forecast results visualized using line plots
- Per-step forecast error plotted to identify degradation over time

How to Use
----------
1. Install requirements:
   pip install tensorflow pandas scikit-learn matplotlib

2. Run the Jupyter Notebook (`main.ipynb`) step-by-step:
   - Preprocess data
   - Train LSTM model
   - Plot and evaluate results

3. (Optional) Use model.predict() for rolling or real-time forecasting.

Author
------
Fadly Syah

Notes
-----
- Consider replacing one-hot route encoding with route embeddings
- You can switch to Temporal Fusion Transformer for improved performance
- For daily forecast only (1-day ahead), set FORECAST_HORIZON = 1
