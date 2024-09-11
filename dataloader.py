'''
def batch_data(X, y, batch_size):
    """Simple function to batch data."""
    for i in range(0, len(X), batch_size):
        X_batch = X.iloc[i:i + batch_size]
        y_batch = y.iloc[i:i + batch_size]
        yield X_batch, y_batch

# Example usage of the batch_data function
if __name__ == "__main__":
    from preprocess import preprocess_data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()
    
    batch_size = 64
    train_batches = list(batch_data(X_train, y_train, batch_size))
    val_batches = list(batch_data(X_val, y_val, batch_size))
    test_batches = list(batch_data(X_test, y_test, batch_size))


import numpy as np

def batch_data(X, y, batch_size):
    """Generate batches of data."""
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        
        yield X_batch, y_batch

# Example usage of the batch_data function
if __name__ == "__main__":
    from preprocess import preprocess_data
    
    # Load preprocessed data
    X_train, X_val, X_test, y_train, y_val, y_test, _, _ = preprocess_data()
    
    # Convert data to NumPy arrays if they are not already
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    
    # Define batch size
    batch_size = 64
    
    # Create batches
    train_batches = list(batch_data(X_train, y_train, batch_size))
    val_batches = list(batch_data(X_val, y_val, batch_size))
    test_batches = list(batch_data(X_test, y_test, batch_size))
    
    # Print example batch to verify
    print("Example training batch:")
    for X_batch, y_batch in train_batches[:1]:  # Print the first batch as an example
        print(X_batch, y_batch)
        break
'''

import numpy as np
import pandas as pd

from preprocess import preprocess_data
'''
def batch_data(X, y, batch_size):
    """Generate batches of data."""
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        
        yield X_batch, y_batch
'''

def batch_data(X, y, batch_size):
    """Yield batches of data."""
    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        if isinstance(X, pd.DataFrame):
            X_batch = X.iloc[batch_indices].values
        else:
            X_batch = X[batch_indices]
        
        y_batch = y[batch_indices]
        
        yield X_batch, y_batch

# Example usage of the batch_data function
if __name__ == "__main__":
    # Load preprocessed data
    X_train, X_val, X_test, y_train, y_val, y_test, _, _ = preprocess_data()
    
    # Ensure data is in NumPy array format
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    
    # Define batch size
    batch_size = 64
    
    # Create batches
    train_batches = list(batch_data(X_train, y_train, batch_size))
    val_batches = list(batch_data(X_val, y_val, batch_size))
    test_batches = list(batch_data(X_test, y_test, batch_size))
    
    # Print example batch to verify
    print("Example training batch:")
    for X_batch, y_batch in train_batches[:1]:  # Print the first batch as an example
        print("X_batch:", X_batch)
        print("y_batch:", y_batch)
        break
