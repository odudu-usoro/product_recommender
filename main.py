from ancf_models import ANCFModel
from preprocess import preprocess_data
from dataloader import batch_data
from train import train_model

def main():
    print("Starting data preprocessing...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()
    print("Data preprocessing completed.")
    
    print("Creating data batches...")
    batch_size = 64
    train_batches = list(batch_data(X_train, y_train, batch_size))
    val_batches = list(batch_data(X_val, y_val, batch_size))
    test_batches = list(batch_data(X_test, y_test, batch_size))
    print(f"Created {len(train_batches)} training batches, {len(val_batches)} validation batches, and {len(test_batches)} test batches.")
    
    print("Instantiating the model...")
    num_users = max(X_train['user_id']) + 1
    num_items = max(X_train['product_id']) + 1
    embedding_dim = 50  # Example embedding dimension
    
    model = ANCFModel(num_users, num_items, embedding_dim)
    print("Model instantiated.")
    
    print("Starting training...")
    epochs = 10
    learning_rate = 0.01
    train_model(model, train_batches, val_batches, epochs, learning_rate)
    print("Training completed.")

if __name__ == "__main__":
    main()
