import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load dataset (assuming it's a CSV file)
# Modify the path accordingly
dataset = pd.read_csv("dataset/wind_power_generated_data.csv")

# Convert DateTime column to datetime type
dataset['DateTime'] = pd.to_datetime(dataset['DateTime'])

# Extract month from DateTime column
dataset['Month'] = dataset['DateTime'].dt.month

# Select relevant columns for X and y
X_columns = ['Month', 'WindSpeed', 'WindDirection', 'AmbientTemperatue', 'BearingShaftTemperature',
             'Blade1PitchAngle', 'Blade2PitchAngle', 'Blade3PitchAngle', 'ControlBoxTemperature',
             'GearboxBearingTemperature', 'GearboxOilTemperature', 'GeneratorRPM',
             'GeneratorWinding1Temperature', 'GeneratorWinding2Temperature', 'HubTemperature',
             'MainBoxTemperature', 'NacellePosition', 'ReactivePower', 'RotorRPM',
             'TurbineStatus']
y_column = 'ActivePowerNext7Days'

# Shift ActivePower to create target variable for the next 7 days
dataset[y_column] = dataset['ActivePower'].shift(-7*24)
dataset.dropna(subset=[y_column], inplace=True)

# Impute missing values in X_columns
dataset[X_columns] = dataset[X_columns].fillna(dataset[X_columns].mean())

# Split dataset into X and y
X = dataset[X_columns].values
y = dataset[y_column].values

# Normalize features using Min-Max Scaler
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)


# Define the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.1)

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Make predictions
predictions = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions - y_test)**2))
print("Root Mean Squared Error (RMSE):", rmse)