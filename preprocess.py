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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data(file_path='amazon.csv'):
    """
    Preprocess the dataset for the ANCF model.

    Args:
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        tuple: Containing X_train, X_val, X_test, y_train, y_val, y_test
    
    Raises:
        FileNotFoundError: If the specified file is not found.
        ValueError: If required columns are missing or contain invalid data.
    """
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Check for required columns
        required_columns = ['rating', 'user_id', 'product_id']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing one or more required columns: {required_columns}")
        
        # Data cleaning
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df = df.dropna(subset=['rating', 'user_id', 'product_id'])
        df = df.drop_duplicates()
        
        if df.empty:
            raise ValueError("After cleaning, the dataset is empty. Check your data.")
        
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
        
        logger.info(f"Train set size: {len(X_train)}")
        logger.info(f"Validation set size: {len(X_val)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except ValueError as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in data preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()
        logger.info("Data preprocessing completed successfully.")
    except Exception as e:
        logger.error(f"Failed to preprocess data: {str(e)}")