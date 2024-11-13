import pandas as pd

# Path to the text file
file_path = 'HEV.txt'

# Load the data into a DataFrame, skipping the first row
df = pd.read_csv(file_path, delim_whitespace=True, names=['Time', 'mph'], skiprows=1)

# Display the first few rows to verify
print(df)