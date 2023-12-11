#Libraries

# General Libraries
import os
import csv
import pandas as pd

#Machine Learning Libraries
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
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

