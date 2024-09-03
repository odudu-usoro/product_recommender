from ancf_models import ANCFModel
from dataloader import batch_data
from preprocess import preprocess_data
from sklearn.metrics import precision_score, recall_score
import numpy as np
# NDCG implementation or use an existing library function

def evaluate_metrics(model, test_batches):
    test_loss = 0
    all_predictions = []
    all_actuals = []
    for X_batch, y_batch in test_batches:
        user_ids = X_batch['user_id'].values
        item_ids = X_batch['product_id'].values
        predictions = model.forward(user_ids, item_ids)
        
        loss = np.mean((predictions - y_batch) ** 2)  # MSE loss
        test_loss += loss
        
        all_predictions.extend(predictions)
        all_actuals.extend(y_batch)

    test_loss = test_loss / len(test_batches)
    precision = precision_score(all_actuals, all_predictions.round())  # Example precision calculation
    recall = recall_score(all_actuals, all_predictions.round())        # Example recall calculation
    ndcg = calculate_ndcg(all_actuals, all_predictions)  # Implement NDCG calculation
    
    return test_loss, precision, recall, ndcg


if __name__ == "__main__":
    # Load and batch the test data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()
    batch_size = 64
    test_batches = list(batch_data(X_test, y_test, batch_size))

   # Instantiate the model (use the maximum user_id and item_id from the entire dataset)
    num_users = max(X_train['user_id'].max(), X_val['user_id'].max(), X_test['user_id'].max()) + 1
    num_items = max(X_train['product_id'].max(), X_val['product_id'].max(), X_test['product_id'].max()) + 1
    embedding_dim = 50

    model = ANCFModel(num_users, num_items, embedding_dim)
    
    # Evaluate on test data
    test_loss, precision, recall, ndcg = evaluate_metrics(model, test_batches)
    print(f"Test Loss: {test_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, NDCG: {ndcg:.4f}")
