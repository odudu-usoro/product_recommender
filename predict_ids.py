import pandas as pd

# Load your dataset
data = pd.read_csv('amazon.csv')  # Adjust the file name as needed

# Extract user and product IDs
user_ids = data['user_id'].unique()
product_ids = data['product_id'].unique()

# Print some sample IDs for prediction
print("Sample User IDs:", user_ids[:5])
print("Sample Product IDs:", product_ids[:5])
