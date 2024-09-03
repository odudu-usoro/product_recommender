from ancf_models import ANCFModel
from preprocess import preprocess_data
from dataloader import batch_data
import numpy as np
import matplotlib.pyplot as plt

def train_model(model, train_batches, val_batches, epochs, learning_rate):
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train_mode()
        train_loss = 0
        
        for X_batch, y_batch in train_batches:
            user_ids = X_batch['user_id'].values
            item_ids = X_batch['product_id'].values
            
            # Forward pass
            predictions = model.forward(user_ids, item_ids)
            loss = np.mean((predictions - y_batch) ** 2)  # MSE loss
            train_loss += loss
            
            # Backward pass
            model.backward(user_ids, item_ids, predictions, y_batch)
        
        avg_train_loss = train_loss / len(train_batches)
        train_losses.append(avg_train_loss)
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}')
        
        # Validate on validation set
        val_loss = validate_model(model, val_batches)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}')
    
    # Plot training and validation loss
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.savefig('training_validation_loss.png')
    #plt.show()

def validate_model(model, val_batches):
    model.eval_mode()
    val_loss = 0
    for X_batch, y_batch in val_batches:
        user_ids = X_batch['user_id'].values
        item_ids = X_batch['product_id'].values
        predictions = model.forward(user_ids, item_ids)
        loss = np.mean((predictions - y_batch) ** 2)  # MSE loss
        val_loss += loss
    return val_loss / len(val_batches)


def main():
    # Data preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()
    
    # Data batching
    batch_size = 64
    train_batches = list(batch_data(X_train, y_train, batch_size))
    val_batches = list(batch_data(X_val, y_val, batch_size))
    test_batches = list(batch_data(X_test, y_test, batch_size))
    
    # Instantiate the model
    num_users = max(X_train['user_id']) + 1
    num_items = max(X_train['product_id']) + 1
    embedding_dim = 50  # Example embedding dimension
    
    model = ANCFModel(num_users, num_items, embedding_dim)
    
    # Define training parameters
    epochs = 10
    learning_rate = 0.01
    
    # Train the model
    train_model(model, train_batches, val_batches, epochs, learning_rate)
    
    # Evaluate the model (optional)
    # from training.eval import evaluate_model
    # ndcg, precision, recall = evaluate_model(model, test_batches)
    # print(f'NDCG: {ndcg}, Precision: {precision}, Recall: {recall}')

if __name__ == "__main__":
    main()