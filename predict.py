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
