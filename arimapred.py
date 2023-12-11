import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Get the current working directory
current_directory = os.getcwd()

# Combine the current directory with the file name
file_name = 'UVT2023.xlsx'
full_file_name = os.path.join(current_directory, file_name)

# Check if the file exists
if os.path.exists(full_file_name):
    df = pd.read_excel(full_file_name)
    # Continue processing the DataFrame

    # Convert date to datetime and set frequency
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    df.index = pd.DatetimeIndex(df.index).to_period('D')  # Setting frequency to 'D' (daily), change as needed

    # Fit ARIMA model
    model = ARIMA(df['UVTMIN'], order=(10, 1, 0))  # Example order (p,d,q) = (10,1,0)
    results = model.fit()

    # Forecast the next 3 steps ahead
    forecast_steps = 3
    last_date = df.index[-1].to_timestamp()  # Convert Period index to Timestamp
    prediction_index = pd.date_range(start=last_date, periods=forecast_steps + 1, closed='right')  # Create a new index for the forecast steps
    forecast = results.forecast(steps=forecast_steps)

    # Append predictions to the DataFrame
    forecast_indexed = pd.Series(forecast, index=prediction_index[1:])  # Exclude the first index to avoid duplicate
    df['ARIMA_Predictions'] = pd.concat([results.fittedvalues, forecast_indexed])

    # Calculate metrics
    actual_values = df['UVTMIN']
    predicted_values = df['ARIMA_Predictions']

    r2 = r2_score(actual_values[:-forecast_steps], predicted_values[:-forecast_steps])  # Adjust to exclude forecast steps from evaluation
    mae = mean_absolute_error(actual_values[:-forecast_steps], predicted_values[:-forecast_steps])
    mse = mean_squared_error(actual_values[:-forecast_steps], predicted_values[:-forecast_steps])
    rmse = mean_squared_error(actual_values[:-forecast_steps], predicted_values[:-forecast_steps], squared=False)

    print(f"R-squared (R2) Score: {r2}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Save the DataFrame with predictions to a CSV file
    df.to_csv('UVT_ARIMA_Predictions.csv')

else:
    print(f"The file '{file_name}' does not exist in the directory '{current_directory}'.")

import pandas as pd

# Assuming 'df' contains 'UVTMIN' and 'ARIMA_Predictions' columns
# Create a DataFrame with 'UVTMIN' and 'ARIMA_Predictions'
uvt_predictions_table = pd.DataFrame({
    'UVTMIN': df['UVTMIN'],
    'ARIMA_Predictions': df['ARIMA_Predictions']  # Replace with your ARIMA predictions column name
})


# Calculate the difference between 'UVTMIN' and 'ARIMA_Predictions'
uvt_predictions_table['Error'] = uvt_predictions_table['UVTMIN'] - uvt_predictions_table['ARIMA_Predictions']

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(uvt_predictions_table['UVTMIN'], uvt_predictions_table['ARIMA_Predictions'])

# Add a column 'Mean Squared Error' with MSE value for all rows
uvt_predictions_table['Mean Squared Error'] = mse

# Display the table
print(uvt_predictions_table)


# Save the table to a CSV file
uvt_predictions_table.to_csv('uvt_predictions_table.csv', index=False)

# Calculate Mean Squared Error (MSE) for each prediction
mse = mean_squared_error(uvt_predictions_table['UVTMIN'], uvt_predictions_table['ARIMA_Predictions'])
print(f"Mean Squared Error (MSE): {mse}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera

# Assuming 'actual_values' are your actual time series values and 'predicted_values' are your model's predictions
# Assuming you have actual_values and predicted_values calculated previously
errors = actual_values - predicted_values

# Convert errors to a list of floats
errors_float = [float(error) for error in errors.astype(float)]

# Error plot over time
plt.figure(figsize=(10, 6))
plt.plot(errors_float)
plt.title('Errors Over Time')
plt.xlabel('Time')
plt.ylabel('Error')
plt.grid(True)
plt.show()



# Histogram of errors
plt.figure(figsize=(8, 6))
plt.hist(errors, bins=30, density=True, alpha=0.7)
plt.title('Histogram of Errors')
plt.xlabel('Error')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# Q-Q plot
sm.qqplot(errors, line='s')
plt.title('Q-Q Plot of Errors')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()

# Calculate descriptive statistics
mean_error = np.mean(errors)
std_error = np.std(errors)
skewness = sm.stats.stattools.skew(errors)
kurtosis = sm.stats.stattools.kurtosis(errors)

print(f"Mean of errors: {mean_error}")
print(f"Standard deviation of errors: {std_error}")
print(f"Skewness of errors: {skewness}")
print(f"Kurtosis of errors: {kurtosis}")

# Ljung-Box test for autocorrelation
lb_test = acorr_ljungbox(errors, lags=[10])
print(f"Ljung-Box test p-value for autocorrelation at lag 10: {lb_test[1][0]}")

# Jarque-Bera test for normality
jb_test = jarque_bera(errors)
print(f"Jarque-Bera test statistic: {jb_test[0]}")
print(f"Jarque-Bera test p-value: {jb_test[1]}")

import numpy as np
import statsmodels.api as sm

# Assuming 'errors' is your list of errors
mean = np.mean(errors)
std_dev = np.std(errors)
skewness = sm.robust.mad(errors) * 1.412

print("Skewness:", skewness)
