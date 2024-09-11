'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data():
    # Load the dataset
    df = pd.read_csv('amazon.csv')
    
    # Data cleaning
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating', 'user_id', 'product_id'])
    df = df.drop_duplicates()
    
    # Encoding
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df['user_id'] = user_encoder.fit_transform(df['user_id'])
    df['product_id'] = item_encoder.fit_transform(df['product_id'])
    
    # Splitting the data
    X = df[['user_id', 'product_id']]
    y = df['rating']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Example of how to call this function in the main script
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data():
    # Load the dataset
    df = pd.read_csv('amazon.csv')
    
    # Data cleaning
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating', 'user_id', 'product_id'])
    df = df.drop_duplicates()
    
    # Encoding
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df['user_id'] = user_encoder.fit_transform(df['user_id'])
    df['product_id'] = item_encoder.fit_transform(df['product_id'])
    
    # Calculate the number of unique users and items
    num_users = len(df['user_id'].unique())
    num_items = len(df['product_id'].unique())
    
    # Verify the encoding
    print("User ID range:", df['user_id'].min(), "to", df['user_id'].max())
    print("Product ID range:", df['product_id'].min(), "to", df['product_id'].max())
    
    # Splitting the data
    X = df[['user_id', 'product_id']]
    y = df['rating']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Convert to NumPy arrays
    X_train_np = np.array([[user_id, item_id] for user_id, item_id in zip(X_train['user_id'], X_train['product_id'])])
    y_train_np = np.array(y_train)
    
    # Verify data shapes
    print("Training data shape:", X_train_np.shape, y_train_np.shape)
    print("Validation data shape:", X_val.shape, y_val.shape)
    print("Test data shape:", X_test.shape, y_test.shape)
    
    # Print number of users and items
    print("Number of unique users:", num_users)
    print("Number of unique items:", num_items)


    # Ensure X_train, X_val, X_test are numpy arrays
    X_train_np = X_train.to_numpy()
    X_val_np = X_val.to_numpy()
    X_test_np = X_test.to_numpy()
    y_train_np = y_train.to_numpy()
    y_val_np = y_val.to_numpy()
    y_test_np = y_test.to_numpy()
    
    num_users = X_train_np[:, 0].max() + 1
    num_items = X_train_np[:, 1].max() + 1
    
    return X_train_np, X_val, X_test, y_train_np, y_val, y_test, num_users, num_items

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, num_users, num_items = preprocess_data()

