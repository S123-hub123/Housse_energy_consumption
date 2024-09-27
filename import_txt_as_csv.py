import pandas as pd

# Define the file path for the .txt file
txt_file_path = '/Users/soukaina/Documents/individual+household+electric+power+consumption/household_power_consumption.txt'

# Read the .txt file as a CSV file (you may need to change the delimiter based on your file's structure)
# For example, if the file is tab-delimited, use '\t'. If comma-separated, use ','.
df = pd.read_csv(txt_file_path, delimiter=';', low_memory=False)

# Display the first few rows of the data (optional)
print(df.head())

# Save the DataFrame as a CSV file
csv_file_path = '/Users/soukaina/Documents/individual+household+electric+power+consumption/household_power_consumption.csv'
df.to_csv(csv_file_path, index=False)

print(f"File saved as {csv_file_path}")
