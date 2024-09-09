'''
from ancf_models import ANCFModel
from dataloader import batch_data
from preprocess import preprocess_data
from sklearn.metrics import mean_squared_error
import numpy as np

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

def evaluate_metrics(model, test_batches):
    all_predictions = []
    all_actuals = []
    for X_batch, y_batch in test_batches:
        user_ids = X_batch['user_id'].values
        item_ids = X_batch['product_id'].values
        predictions = model.forward(user_ids, item_ids)
        
        all_predictions.extend(predictions)
        all_actuals.extend(y_batch)

    # Convert lists to NumPy arrays
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)

    mse = mean_squared_error(all_actuals, all_predictions)
    ndcg = calculate_ndcg(all_actuals, all_predictions)
    
    return mse, ndcg

if __name__ == "__main__":
    # Load and batch the test data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()
    batch_size = 64
    test_batches = list(batch_data(X_test, y_test, batch_size))

    # Instantiate the model (use the maximum user_id and item_id from the entire dataset)
    num_users = max(X_train['user_id'].max(), X_val['user_id'].max(), X_test['user_id'].max()) + 1
    num_items = max(X_train['product_id'].max(), X_val['product_id'].max(), X_test['product_id'].max()) + 1
    embedding_dim = 100

    model = ANCFModel(num_users, num_items, embedding_dim)
    
    # Evaluate on test data
    mse, ndcg = evaluate_metrics(model, test_batches)
    print(f"Test MSE: {mse:.4f}, NDCG: {ndcg:.4f}")


from ancf_models import ANCFModel
from dataloader import batch_data
from preprocess import preprocess_data
from sklearn.metrics import mean_squared_error
import numpy as np

def dcg_score(y_true, y_score, k=10):
    """Compute DCG@k for a single sample."""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def calculate_ndcg(y_true, y_score, k=10):
    """Compute NDCG@k for a batch of samples."""
    actual_dcg = dcg_score(y_true, y_score, k)
    best_dcg = dcg_score(y_true, y_true, k)
    return actual_dcg / best_dcg if best_dcg > 0 else 0

def evaluate_model(model, test_batches):
    y_true_all = []
    y_pred_all = []

    for X_batch, y_batch in test_batches:
        user_ids = X_batch['user_id'].values
        item_ids = X_batch['product_id'].values
        predictions = model.forward(user_ids, item_ids)

        y_true_all.extend(y_batch)
        y_pred_all.extend(predictions)

    # Compute MSE and NDCG
    mse = mean_squared_error(y_true_all, y_pred_all)
    ndcg = calculate_ndcg(np.array(y_true_all), np.array(y_pred_all))

    print(f"Test MSE: {mse:.4f}")
    print(f"Test NDCG: {ndcg:.4f}")

def main():
    # Data preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()

    # Data batching
    batch_size = 64
    test_batches = list(batch_data(X_test, y_test, batch_size))

    # Instantiate the model (ensure same parameters used in training)
    num_users = max(X_train['user_id']) + 1
    num_items = max(X_train['product_id']) + 1
    embedding_dim = 80

    model = ANCFModel(num_users, num_items, embedding_dim)

    # Evaluate the model
    evaluate_model(model, test_batches)

if __name__ == "__main__":
    main()


import tensorflow as tf
from ancf_models import ANCFModel
from dataloader import batch_data
from preprocess import preprocess_data
import numpy as np

def dcg_score(y_true, y_score, k=10):
    """Compute DCG@k for a single sample."""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def calculate_ndcg(y_true, y_score, k=10):
    """Compute NDCG@k for a batch of samples."""
    actual_dcg = dcg_score(y_true, y_score, k)
    best_dcg = dcg_score(y_true, y_true, k)
    return actual_dcg / best_dcg if best_dcg > 0 else 0

def evaluate_model(model, test_batches):
    y_true_all = []
    y_pred_all = []

    for X_batch, y_batch in test_batches:
        user_ids = X_batch['user_id'].values
        item_ids = X_batch['product_id'].values

        # Convert to TensorFlow tensors
        user_ids = tf.convert_to_tensor(user_ids, dtype=tf.int32)
        item_ids = tf.convert_to_tensor(item_ids, dtype=tf.int32)

        # Make predictions
        predictions = model(user_ids, item_ids).numpy()

        y_true_all.extend(y_batch)
        y_pred_all.extend(predictions)

    # Compute MSE and NDCG
    mse = np.mean(np.square(np.array(y_true_all) - np.array(y_pred_all)))
    ndcg = calculate_ndcg(np.array(y_true_all), np.array(y_pred_all))

    print(f"Test MSE: {mse:.4f}")
    print(f"Test NDCG: {ndcg:.4f}")

def main():
    # Data preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()

    # Data batching
    batch_size = 64
    test_batches = list(batch_data(X_test, y_test, batch_size))

    # Instantiate the model (ensure same parameters used in training)
    num_users = max(X_train['user_id']) + 1
    num_items = max(X_train['product_id']) + 1
    embedding_dim = 80

    model = ANCFModel(num_users, num_items, embedding_dim)

    # Load the trained model weights if saved
    # model.load_weights('path_to_saved_model_weights')

    # Evaluate the model
    evaluate_model(model, test_batches)

if __name__ == "__main__":
    main()
'''

import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error, precision_score, recall_score, average_precision_score
from ancf_models import ANCFModel
from dataloader import batch_data
from preprocess import preprocess_data

def dcg_score(y_true, y_score, k=10):
    """Compute DCG@k for a single sample."""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def calculate_ndcg(y_true, y_score, k=10):
    """Compute NDCG@k for a batch of samples."""
    actual_dcg = dcg_score(y_true, y_score, k)
    best_dcg = dcg_score(y_true, y_true, k)
    return actual_dcg / best_dcg if best_dcg > 0 else 0

def precision_at_k(y_true, y_score, k=10):
    """Compute Precision@k."""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.mean(y_true > 0)

def recall_at_k(y_true, y_score, k=10):
    """Compute Recall@k."""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.sum(y_true > 0) / np.sum(y_true)

def evaluate_model(model, test_batches):
    y_true_all = []
    y_pred_all = []
    
    for X_batch, y_batch in test_batches:
        user_ids = tf.convert_to_tensor(X_batch['user_id'].values, dtype=tf.int32)
        item_ids = tf.convert_to_tensor(X_batch['product_id'].values, dtype=tf.int32)
        
        predictions = model(user_ids, item_ids).numpy().flatten()
        y_true_all.extend(y_batch)
        y_pred_all.extend(predictions)
    
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    
    # Compute MSE
    mse = mean_squared_error(y_true_all, y_pred_all)
    
    # Compute NDCG@10
    ndcg = calculate_ndcg(y_true_all, y_pred_all, k=10)
    
    # Compute Precision@10 and Recall@10
    precision = precision_at_k(y_true_all, y_pred_all, k=10)
    recall = recall_at_k(y_true_all, y_pred_all, k=10)
    
    # Compute Mean Average Precision
    map_score = average_precision_score(y_true_all > 0, y_pred_all)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test NDCG@10: {ndcg:.4f}")
    print(f"Test Precision@10: {precision:.4f}")
    print(f"Test Recall@10: {recall:.4f}")
    print(f"Test MAP: {map_score:.4f}")

def main():
    # Data preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()
    
    # Data batching
    batch_size = 64
    test_batches = list(batch_data(X_test, y_test, batch_size))
    
    # Instantiate the model (ensure same parameters used in training)
    num_users = max(X_train['user_id'].max(), X_test['user_id'].max()) + 1
    num_items = max(X_train['product_id'].max(), X_test['product_id'].max()) + 1
    embedding_dim = 80
    model = ANCFModel(num_users, num_items, embedding_dim)