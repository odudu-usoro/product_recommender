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
