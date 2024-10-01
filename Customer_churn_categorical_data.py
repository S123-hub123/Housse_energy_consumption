import pandas as pd

# Data collection from CSV (assuming it's stored locally or fetched from an API)
data = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/telco-customer-churn.csv')

# Check the size of the dataset
print(f"Data shape: {data.shape}")
