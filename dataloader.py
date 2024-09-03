#import numpy as np

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
