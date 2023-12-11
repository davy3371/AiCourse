#Libraries

# General Libraries
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Machine Learning Libraries - SVR - Sci-Kit Learn

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split



# Get the current working directory
current_directory = os.getcwd()

# Combine the current directory with the file name
file_name = 'UVT2023.xlsx'
full_file_name = os.path.join(current_directory, file_name)

# Check if the file exists
if os.path.exists(full_file_name):
    df = pd.read_excel(full_file_name)
    # Continue processing the DataFrame
else:
    print(f"The file '{file_name}' does not exist in the directory '{current_directory}'.")



# Read the Excel file into a Pandas DataFrame
df = pd.read_excel(full_file_name)

# Convert date to datetime
print(df.columns) 
df['DATE'] = pd.to_datetime(df['DATE'])
print("datetime worked")





# Display the DataFrame or perform operations on it
print(df.head())  # Display the first few rows of the DataFrame

# Check for missing values in the entire DataFrame
missing_values = df.isna().sum()

# Display columns with missing values and their count
print("Columns with missing values:")
#print("None"[missing_values = 0])
print(missing_values[missing_values >= 0])

# fill any missing values with zero 0
df = df.fillna(0)
print("any missing values replaced by zero")


# Graph average uvt over time
plt.figure(figsize=(10, 6))
plt.plot(df['DATE'], df['UVTMIN'], marker='o')
plt.title('UVT Min Over Time')
plt.xlabel('DATE')
plt.ylabel('UVT Min')
plt.grid(True)
plt.show()
print("UVT MIN graph over time")



# Assuming df is already loaded with columns 'DATE' and 'UVTMIN'

# Convert the 'DATE' column to datetime 
df['DATE'] = pd.to_datetime(df['DATE'])
print("date to datetime")
print("new columns are:")
print(df.columns)

# Set 'date' column as the index but leave date column intact
df.set_index('DATE', inplace=True, drop=False)
print("set index")
print("new columns are:")
print(df.columns)

# Plot the time series
df['UVTMIN'].plot(figsize=(12, 6))
plt.title('Time Series of UVT')
plt.xlabel('Date')
plt.ylabel('UVT AVG')
plt.show()
print("graph")

# Fit SVR from dataframe with columns DATE and UVTMIN

print("new columns are:")
print(df.columns)

# Scale the UVTMIN data
scaler = StandardScaler()
df['UVT_scaled'] = scaler.fit_transform(df['UVTMIN'].values.reshape(-1, 1))
print("new columns are:")
print(df.columns)


# Create SVR model and fit it to the data
svr = SVR(kernel='rbf', C=100, gamma='auto', epsilon=0.1)
svr.fit(df['DATE'].values.reshape(-1, 1), df['UVT_scaled'])

# Predict using the trained SVR model
predictions_scaled = svr.predict(df['DATE'].values.reshape(-1, 1))

# Inverse scaling for predictions
df['Predictions'] = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))

# Plot the original data and SVR predictions
plt.figure(figsize=(10, 6))
plt.scatter(df['DATE'], df['UVTMIN'], color='blue', label='Original data')
plt.plot(df['DATE'], df['Predictions'], color='red', label='SVR predictions')
plt.xlabel('DATE')
plt.ylabel('UVTMIN')
plt.title('SVR for Time Series Prediction')
plt.legend()
plt.show()

# Assuming 'df' contains 'UVTMIN' and 'Predictions' columns
# Create a new DataFrame containing 'UVTMIN' and 'Predictions' columns
uvt_predictions_table = pd.DataFrame({
    'UVTMIN': df['UVTMIN'],
    'SVR_Predictions': df['Predictions']
})

# Display the table
print(uvt_predictions_table)


# Assuming 'df' contains 'UVTMIN' and 'Predictions' columns
actual_values = df['UVTMIN']
predicted_values = df['Predictions']

# Calculate R2 score
r2 = r2_score(actual_values, predicted_values)

# Calculate MAE
mae = mean_absolute_error(actual_values, predicted_values)

# Calculate MSE
mse = mean_squared_error(actual_values, predicted_values)

# Calculate RMSE
rmse = mean_squared_error(actual_values, predicted_values, squared=False)

print(f"R-squared (R2) Score: {r2}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Assuming 'df' contains your DataFrame with 'DATE', 'UVTMIN', and 'Predictions' columns
