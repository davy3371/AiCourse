#Libraries

# General Libraries
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd

#Machine Learning Libraries - ARIMA

from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error




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



# Assuming df is already loaded with columns 'DATE' and 'uvt'

# Convert the 'DATE' column to datetime if it's not already in datetime format
df['DATE'] = pd.to_datetime(df['DATE'])
print("date to datetime")


# Set 'date' column as the index
df.set_index('DATE', inplace=True)
print("set index")

# Plot the time series
df['UVTMIN'].plot(figsize=(12, 6))
plt.title('Time Series of UVT')
plt.xlabel('Date')
plt.ylabel('UVT AVG')
plt.show()
print("graph")

# Fit ARIMA model
model = ARIMA(df['UVTMIN'], order=(10,1,0))  # Example order (p,d,q) = (5,1,0)
results = model.fit()
print("model fit")

# Summary of the model
print(results)
print(results.summary())

# Forecast the next 10 steps ahead
forecast_steps = 3
forecast = results.forecast(steps=forecast_steps)

# Print forecasted values
print(forecast)

# Assuming you have fitted an ARIMA model and obtained predictions
# Replace 'actual_values' and 'predicted_values' with your actual and predicted values


# Calculate metrics
#mae = mean_absolute_error(actual_values, predicted_values)
#mse = mean_squared_error(actual_values, predicted_values)
#rmse = mean_squared_error(actual_values, predicted_values, squared=False)  # RMSE
#mape = mean_absolute_percentage_error(actual_values, predicted_values)

#print(f"MAE: {mae}")
#print(f"MSE: {mse}")
#print(f"RMSE: {rmse}")
#print(f"MAPE: {mape}")


# Assuming you have a results object from statsmodels
# Replace this with your actual results object
#results = sm.OLS(y, X).fit()

# Get the summary as a string
#summary_text = results.summary()

# Split the summary into lines and extract the table
#summary_lines = summary_text.split('\n')
#table_data = [line.split() for line in summary_lines if line.strip() != '']

# Convert the table data to a pandas DataFrame
#df = pd.DataFrame(table_data[1:], columns=table_data[0])

# Save the DataFrame to a CSV file
#df.to_csv('ARIMA_results.csv', index=False)
