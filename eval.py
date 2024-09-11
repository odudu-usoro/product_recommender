'''
from ancf_models import ANCFModel
from dataloader import batch_data
from preprocess import preprocess_data
from sklearn.metrics import mean_squared_error
import numpy as np

# Existing DCG and NDCG functions
def dcg_score(y_true, y_score, k=10):
    """Compute DCG@k for a single sample."""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def calculate_ndcg(y_true, y_score, k=10):
    """Compute NDCG@k for all samples."""
    actual_dcg = dcg_score(y_true, y_score, k)
    best_dcg = dcg_score(y_true, y_true, k)
    return actual_dcg / best_dcg if best_dcg > 0 else 0

# NEW: Precision@K and Recall@K functions added here
def precision_at_k(y_true, y_score, k=10):
    """Compute Precision@k for all samples."""
    order = np.argsort(y_score)[::-1]
    top_k_true = np.take(y_true, order[:k])
    return np.mean(top_k_true)

def recall_at_k(y_true, y_score, k=10):
    """Compute Recall@k for all samples."""
    order = np.argsort(y_score)[::-1]
    top_k_true = np.take(y_true, order[:k])
    relevant_items = np.sum(y_true)
    return np.sum(top_k_true) / relevant_items if relevant_items > 0 else 0

# Modified evaluate_metrics function to include Precision@K and Recall@K
def evaluate_metrics(model, test_batches, k=10):
    all_predictions = []
    all_actuals = []
    for X_batch, y_batch in test_batches:
        user_ids = X_batch['user_id'].values
        item_ids = X_batch['product_id'].values
        predictions = model.predict(user_ids, item_ids)  # Use predict method
        
        all_predictions.extend(predictions)
        all_actuals.extend(y_batch)

    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)

    mse = mean_squared_error(all_actuals, all_predictions)
    ndcg = calculate_ndcg(all_actuals, all_predictions, k)
    precision = precision_at_k(all_actuals, all_predictions, k)
    recall = recall_at_k(all_actuals, all_predictions, k)
    
    return mse, ndcg, precision, recall

if __name__ == "__main__":
    # Load and batch the test data
    X_train, X_val, X_test, y_train, y_val, y_test, num_users, num_items = preprocess_data()
    batch_size = 64
    test_batches = list(batch_data(X_test, y_test, batch_size))

    # Instantiate the model (use the maximum user_id and item_id from the entire dataset)
    num_users = max(X_train['user_id'].max(), X_val['user_id'].max(), X_test['user_id'].max()) + 1
    num_items = max(X_train['product_id'].max(), X_val['product_id'].max(), X_test['product_id'].max()) + 1
    embedding_dim = 100

    model = ANCFModel(num_users, num_items, embedding_dim)
    
    # Evaluate on test data
    mse, ndcg, precision, recall = evaluate_metrics(model, test_batches)  # Updated call
    print(f"Test MSE: {mse:.4f}, NDCG: {ndcg:.4f}")
    print(f"Precision@K: {precision:.4f}, Recall@K: {recall:.4f}")  # NEW: Print additional metrics

    user_ids = X_test['user_id'].values
    item_ids = X_test['product_id'].values
    predictions = model.predict(user_ids, item_ids)
    print(f'Predictions: {predictions}')


from ancf_models import ANCFModel
from dataloader import batch_data
from preprocess import preprocess_data
from sklearn.metrics import mean_squared_error
import numpy as np
import os

# Existing DCG and NDCG functions
def dcg_score(y_true, y_score, k=10):
    """Compute DCG@k for a single sample."""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def calculate_ndcg(y_true, y_score, k=10):
    """Compute NDCG@k for all samples."""
    actual_dcg = dcg_score(y_true, y_score, k)
    best_dcg = dcg_score(y_true, y_true, k)
    return actual_dcg / best_dcg if best_dcg > 0 else 0

# Precision@K and Recall@K functions
def precision_at_k(y_true, y_score, k=10):
    """Compute Precision@k for all samples."""
    order = np.argsort(y_score)[::-1]
    top_k_true = np.take(y_true, order[:k])
    return np.mean(top_k_true)

def recall_at_k(y_true, y_score, k=10):
    """Compute Recall@k for all samples."""
    order = np.argsort(y_score)[::-1]
    top_k_true = np.take(y_true, order[:k])
    relevant_items = np.sum(y_true)
    return np.sum(top_k_true) / relevant_items if relevant_items > 0 else 0

# Evaluate metrics function
def evaluate_metrics(model, test_batches, k=10):
    all_predictions = []
    all_actuals = []
    for X_batch, y_batch in test_batches:
        user_ids = X_batch[:, 0]
        item_ids = X_batch[:, 1]
        predictions = model.forward(user_ids, item_ids)  # Use forward method
        
        all_predictions.extend(predictions)
        all_actuals.extend(y_batch)

    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)

    mse = mean_squared_error(all_actuals, all_predictions)
    ndcg = calculate_ndcg(all_actuals, all_predictions, k)
    precision = precision_at_k(all_actuals, all_predictions, k)
    recall = recall_at_k(all_actuals, all_predictions, k)
    
    return mse, ndcg, precision, recall

if __name__ == "__main__":
    # Load and batch the test data
    X_train, X_val, X_test, y_train, y_val, y_test, num_users, num_items = preprocess_data()

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    batch_size = 64
    test_batches = list(batch_data(X_test, y_test, batch_size))
    
    # Instantiate the model with the same parameters used in training
    embedding_dim = 80  # Use the embedding dimension from training
    model = ANCFModel(num_users, num_items, embedding_dim)
    
    # Load the best model checkpoint
    checkpoint_path = 'best_model.npy'
    if os.path.exists(checkpoint_path):
        checkpoint = np.load(checkpoint_path, allow_pickle=True).item()
        model.user_embedding = checkpoint['user_embedding']
        model.item_embedding = checkpoint['item_embedding']
        model.weights = checkpoint['weights']
        model.bias = checkpoint['bias']
        print(f"Checkpoint loaded from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
    
    # Evaluate on test data
    mse, ndcg, precision, recall = evaluate_metrics(model, test_batches)  # Updated call
    print(f"Test MSE: {mse:.4f}, NDCG: {ndcg:.4f}")
    print(f"Precision@K: {precision:.4f}, Recall@K: {recall:.4f}")
'''

from ancf_models import ANCFModel
from dataloader import batch_data
from preprocess import preprocess_data
from sklearn.metrics import mean_squared_error
import numpy as np
import os

# Existing DCG and NDCG functions
def dcg_score(y_true, y_score, k=10):
    """Compute DCG@k for a single sample."""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def calculate_ndcg(y_true, y_score, k=10):
    """Compute NDCG@k for all samples."""
    actual_dcg = dcg_score(y_true, y_score, k)
    best_dcg = dcg_score(y_true, y_true, k)
    return actual_dcg / best_dcg if best_dcg > 0 else 0

# Precision@K and Recall@K functions
def precision_at_k(y_true, y_score, k=10):
    """Compute Precision@k for all samples."""
    order = np.argsort(y_score)[::-1]
    top_k_true = np.take(y_true, order[:k])
    return np.mean(top_k_true)

def recall_at_k(y_true, y_score, k=10):
    """Compute Recall@k for all samples."""
    order = np.argsort(y_score)[::-1]
    top_k_true = np.take(y_true, order[:k])
    relevant_items = np.sum(y_true)
    return np.sum(top_k_true) / relevant_items if relevant_items > 0 else 0

# Evaluate metrics function
def evaluate_metrics(model, test_batches, k=10):
    all_predictions = []
    all_actuals = []
    for X_batch, y_batch in test_batches:
        user_ids = X_batch[:, 0]
        item_ids = X_batch[:, 1]
        predictions = model.forward(user_ids, item_ids)  # Use forward method
        
        all_predictions.extend(predictions)
        all_actuals.extend(y_batch)

    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)

    mse = mean_squared_error(all_actuals, all_predictions)
    ndcg = calculate_ndcg(all_actuals, all_predictions, k)
    precision = precision_at_k(all_actuals, all_predictions, k)
    recall = recall_at_k(all_actuals, all_predictions, k)
    
    return mse, ndcg, precision, recall

if __name__ == "__main__":
    # Load and batch the test data
    X_train, X_val, X_test, y_train, y_val, y_test, num_users, num_items = preprocess_data()

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    batch_size = 64
    test_batches = list(batch_data(X_test, y_test, batch_size))
    
    # Instantiate the model with the same parameters used in training
    embedding_dim = 80  # Use the embedding dimension from training
    model = ANCFModel(num_users, num_items, embedding_dim)
    
    # Load the best model checkpoint
    checkpoint_path = 'best_model.npy'
    if os.path.exists(checkpoint_path):
        model.load_checkpoint(checkpoint_path)
    
    # Evaluate the model
    mse, ndcg, precision, recall = evaluate_metrics(model, test_batches)
    print(f"MSE: {mse:.4f}")
    print(f"NDCG@10: {ndcg:.4f}")
    print(f"Precision@10: {precision:.4f}")
    print(f"Recall@10: {recall:.4f}")
