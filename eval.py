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


from ancf_models import ANCFModelAttention
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
'''

import numpy as np
from sklearn.metrics import mean_squared_error
from preprocess import preprocess_data
from ancf_models import ANCFModelAttention

# ===============================================================
# Ranking metric functions (all pure NumPy)
# ===============================================================
def dcg_at_k(r, k):
    """Discounted Cumulative Gain at rank k."""
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.0

def ndcg_at_k(r, k):
    """Normalized DCG at rank k."""
    ideal = sorted(r, reverse=True)
    best_dcg = dcg_at_k(ideal, k)
    return dcg_at_k(r, k) / best_dcg if best_dcg > 0 else 0.0

def compute_ndcg_at_k(user_to_preds, user_to_true, k):
    ndcgs = []
    for user in user_to_preds:
        preds = sorted(user_to_preds[user], key=lambda x: x[1], reverse=True)
        true_dict = dict(user_to_true.get(user, []))
        rels = [true_dict.get(i, 0) for i, _ in preds]
        ndcgs.append(ndcg_at_k(rels, k))
    return np.mean(ndcgs) if ndcgs else 0.0

def compute_precision_at_k(user_to_preds, user_to_true, k):
    precisions = []
    for user in user_to_preds:
        preds = sorted(user_to_preds[user], key=lambda x: x[1], reverse=True)[:k]
        true_dict = dict(user_to_true.get(user, []))
        top_k_items = [i for i, _ in preds]
        hits = sum(1 for i in top_k_items if true_dict.get(i, 0) > 0)
        precisions.append(hits / k)
    return np.mean(precisions) if precisions else 0.0

def compute_recall_at_k(user_to_preds, user_to_true, k):
    recalls = []
    for user in user_to_preds:
        preds = sorted(user_to_preds[user], key=lambda x: x[1], reverse=True)[:k]
        true_dict = dict(user_to_true.get(user, []))
        relevant_items = [i for i, v in true_dict.items() if v > 0]
        if not relevant_items:
            continue
        top_k_items = [i for i, _ in preds]
        hits = sum(1 for i in top_k_items if i in relevant_items)
        recalls.append(hits / len(relevant_items))
    return np.mean(recalls) if recalls else 0.0


# ===============================================================
# Evaluation function
# ===============================================================
def evaluate_metrics(model, X_test, y_test, user_histories_dict, k=10):
    preds, targets = [], []
    user_to_preds = {}
    user_to_true = {}

    for (user_id, item_id), true_rating in zip(X_test.to_numpy(), y_test.to_numpy()):
        # --- Manual forward pass (same as in training) ---
        user_emb = model.user_embeddings[user_id]
        hist = user_histories_dict.get(user_id, [])
        hist_embs = model.item_embeddings[hist] if len(hist) > 0 else np.zeros((1, model.embedding_dim))

        Q = user_emb @ model.Wq
        K = hist_embs @ model.Wk
        V = hist_embs @ model.Wv
        attn_scores = (Q @ K.T) / np.sqrt(model.embedding_dim)
        attn_weights = np.exp(attn_scores - attn_scores.max())
        attn_weights /= attn_weights.sum()
        context = attn_weights @ V
        x_vec = np.concatenate([user_emb, context])
        pred = x_vec @ model.W_out + model.b_out
        pred = float(pred.item() if np.ndim(pred) > 0 else pred)

        preds.append(pred)
        targets.append(true_rating)

        # For ranking metrics
        user_to_preds.setdefault(user_id, []).append((item_id, pred))
        # Define a binary relevance threshold — adjust based on your dataset
        is_relevant = 1 if true_rating >= 4 else 0   # If ratings are on a 1–5 scale
        user_to_true.setdefault(user_id, []).append((item_id, is_relevant))

        print(sum(len(v) for v in user_to_true.values()), "test interactions total")
        print(sum(sum(r for _, r in v) for v in user_to_true.values()), "positive interactions total")


    # Compute metrics
    mse = mean_squared_error(targets, preds)
    ndcg = compute_ndcg_at_k(user_to_preds, user_to_true, k)
    precision = compute_precision_at_k(user_to_preds, user_to_true, k)
    recall = compute_recall_at_k(user_to_preds, user_to_true, k)

    print("\n📊 Evaluation Results")
    print(f"  • MSE: {mse:.4f} — Lower is better (average squared error)")
    print(f"  • NDCG@{k}: {ndcg:.4f} — Ranking quality; higher = better")
    print(f"  • Precision@{k}: {precision:.4f} — Fraction of top-{k} that are relevant")
    print(f"  • Recall@{k}: {recall:.4f} — Fraction of relevant items found in top-{k}")
    print("")

    return mse, ndcg, precision, recall


# ===============================================================
# Main script
# ===============================================================
if __name__ == "__main__":
    print("Running preprocessing...")
    X_train, X_val, X_test, y_train, y_val, y_test, num_users, num_items = preprocess_data()

    # Build user histories
    user_histories = {}
    for u, i in zip(X_train["user_id"], X_train["product_id"]):
        user_histories.setdefault(u, []).append(i)

    num_users = max(X_train["user_id"].max(), X_val["user_id"].max(), X_test["user_id"].max()) + 1
    num_items = max(X_train["product_id"].max(), X_val["product_id"].max(), X_test["product_id"].max()) + 1

    model = ANCFModelAttention(num_users=num_users, num_items=num_items, embedding_dim=64)

    checkpoint_path = "best_model_attention.npy"
    model.load_checkpoint(checkpoint_path)
    print("✅ Loaded best model checkpoint.")

    mse, ndcg, precision, recall = evaluate_metrics(model, X_test, y_test, user_histories, k=10)
