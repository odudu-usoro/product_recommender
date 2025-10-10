'''
import pandas as pd
import numpy as np
import os
from ancf_models import ANCFModel

def load_dataset(file_path):
    """Load dataset and return user_ids and product_ids."""
    df = pd.read_csv(file_path)
    
    # Convert IDs to integer encodings
    user_ids, user_id_mapping = pd.factorize(df['user_id'])
    product_ids, product_id_mapping = pd.factorize(df['product_id'])
    
    return user_ids, product_ids, user_id_mapping, product_id_mapping

def predict_ratings(model, user_ids, product_ids):
    """Predict ratings for given user_ids and product_ids using the model."""
    predictions = model.predict(user_ids, product_ids)
    return predictions

def main():
    # Paths
    dataset_path = 'amazon.csv'
    checkpoint_path = 'best_model.npy'
    
    # Load dataset
    user_ids, product_ids, user_id_mapping, product_id_mapping = load_dataset(dataset_path)
    
    # Determine number of unique users and items
    num_users = len(user_id_mapping)
    num_items = len(product_id_mapping)
    
    # Load model
    embedding_dim = 80  # Make sure this matches the model's training configuration
    model = ANCFModel(num_users, num_items, embedding_dim)
    
    if os.path.exists(checkpoint_path):
        checkpoint = np.load(checkpoint_path, allow_pickle=True).item()
        model.user_embedding = checkpoint['user_embedding']
        model.item_embedding = checkpoint['item_embedding']
        model.weights = checkpoint['weights']
        model.bias = checkpoint['bias']
        print(f"Checkpoint loaded from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")

    # Predict ratings
    predictions = predict_ratings(model, user_ids, product_ids)
    
    # Display or save predictions
    for user_id, product_id, prediction in zip(user_ids, product_ids, predictions):
        print(f"User ID: {user_id}, Product ID: {product_id}, Predicted Rating: {prediction:.2f}")
    
    # Optionally, save predictions to a CSV file
    results_df = pd.DataFrame({
        'user_id': user_ids,
        'product_id': product_ids,
        'predicted_rating': predictions
    })
    results_df.to_csv('predictions.csv', index=False)
    print("Predictions saved to predictions.csv")

if __name__ == "__main__":
    main()
'''

import numpy as np
import pandas as pd
from ancf_models import ANCFModelAttention
from preprocess import preprocess_data

def predict_rating(model, user_id, item_id, user_histories):
    """Compute predicted rating for a single user–item pair using attention."""
    user_emb = model.user_embeddings[user_id]
    hist = user_histories.get(user_id, [])
    hist_embs = model.item_embeddings[hist] if len(hist) > 0 else np.zeros((1, model.embedding_dim))

    # Attention mechanism
    Q = user_emb @ model.Wq
    K = hist_embs @ model.Wk
    V = hist_embs @ model.Wv
    attn_scores = (Q @ K.T) / np.sqrt(model.embedding_dim)
    attn_weights = np.exp(attn_scores - attn_scores.max())
    attn_weights /= attn_weights.sum()
    context = attn_weights @ V

    # Final prediction
    x_vec = np.concatenate([user_emb, context])
    pred = x_vec @ model.W_out + model.b_out
    return float(pred)

def load_best_model(checkpoint_path):
    """Load the best saved model checkpoint."""
    checkpoint = np.load(checkpoint_path, allow_pickle=True).item()
    model = ANCFModelAttention(
        num_users=checkpoint['user_embeddings'].shape[0],
        num_items=checkpoint['item_embeddings'].shape[0],
        embedding_dim=checkpoint['user_embeddings'].shape[1]
    )
    model.__dict__.update(checkpoint)
    print(f"✅ Loaded model from {checkpoint_path}")
    return model

def build_user_histories(X_train_df):
    """Build user history dictionary from training data."""
    user_histories = {}
    for u, i in zip(X_train_df["user_id"], X_train_df["product_id"]):
        if u not in user_histories:
            user_histories[u] = []
        user_histories[u].append(i)
    return user_histories

def generate_topN_recommendations(model, user_histories, num_users, num_items, top_n=10):
    """Generate top-N product recommendations per user."""
    all_recs = {}
    for user_id in range(num_users):
        preds = np.zeros(num_items)
        for item_id in range(num_items):
            preds[item_id] = predict_rating(model, user_id, item_id, user_histories)
        top_items = np.argsort(preds)[::-1][:top_n]
        all_recs[user_id] = top_items
    return all_recs

if __name__ == "__main__":
    print("🔄 Loading data and model...")
    X_train, X_val, X_test, y_train, y_val, y_test, num_users, num_items, user_histories = preprocess_data(return_histories=True)
    model = load_best_model("best_model_attention.npy")

    num_users = model.user_embeddings.shape[0]
    num_items = model.item_embeddings.shape[0]

    print("\n🎯 Generating Top-10 recommendations per user...")
    topN = generate_topN_recommendations(model, user_histories, num_users, num_items, top_n=10)

    # Convert to DataFrame
    recs_df = pd.DataFrame([
        {"user_id": user, "recommended_items": topN[user].tolist()}
        for user in topN
    ])

    # Save to CSV
    recs_df.to_csv("top10_recommendations.csv", index=False)
    print("✅ Top-10 recommendations saved to top10_recommendations.csv")

    # Show some samples
    print("\n📋 Sample Recommendations:")
    for user in list(topN.keys())[:5]:
        print(f"User {user} → Top-10 Items: {topN[user].tolist()}")
