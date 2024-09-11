'''
from ancf_models import ANCFModel
from preprocess import preprocess_data
from dataloader import batch_data
import numpy as np
import matplotlib.pyplot as plt
import os

def train_model(model, train_batches, val_batches, epochs, initial_lr, patience=3, decay_factor=0.9, checkpoint_path='best_model.npy'):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    learning_rate = initial_lr
    
    for epoch in range(epochs):
        # model.train_mode()  # Switch to training mode
        train_loss = 0
        
        for X_batch, y_batch in train_batches:
            user_ids = X_batch['user_id'].values
            item_ids = X_batch['product_id'].values
            
            # Forward pass
            predictions = model.forward(user_ids, item_ids)
            loss = np.mean((predictions - y_batch) ** 2)  # MSE loss
            train_loss += loss
            
            # Backward pass
            model.learning_rate = learning_rate  # Apply the dynamic learning rate
            model.backward(user_ids, item_ids, predictions, y_batch)
        
        avg_train_loss = train_loss / len(train_batches)
        train_losses.append(avg_train_loss)
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}')
        
        # Validate on validation set
        val_loss = validate_model(model, val_batches)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}')
        
        # Save model checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, checkpoint_path)
            print(f"Model checkpoint saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
        
        # Apply learning rate decay after each epoch
        learning_rate *= decay_factor
        print(f"Epoch {epoch+1}, Learning Rate: {learning_rate}")

        # Inside the training loop
        print("Epoch:", epoch + 1)
        print("User Embeddings:", model.user_embedding)
        print("Item Embeddings:", model.item_embedding)

    
    # Plot training and validation loss
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.savefig('training_validation_loss.png')

def validate_model(model, val_batches):
    # model.eval_mode()  # Switch to evaluation mode
    val_loss = 0
    for X_batch, y_batch in val_batches:
        user_ids = X_batch['user_id'].values
        item_ids = X_batch['product_id'].values
        predictions = model.forward(user_ids, item_ids)
        loss = np.mean((predictions - y_batch) ** 2)  # MSE loss
        val_loss += loss
    return val_loss / len(val_batches)

def save_checkpoint(model, checkpoint_path):
    """Save the model state to a file."""
    np.save(checkpoint_path, {
        'user_embedding': model.user_embedding,
        'item_embedding': model.item_embedding,
        'weights': model.weights,
        'bias': model.bias
    })

def load_checkpoint(model, checkpoint_path):
    """Load the model state from a file."""
    if os.path.exists(checkpoint_path):
        checkpoint = np.load(checkpoint_path, allow_pickle=True).item()
        model.user_embedding = checkpoint['user_embedding']
        model.item_embedding = checkpoint['item_embedding']
        model.weights = checkpoint['weights']
        model.bias = checkpoint['bias']
        print(f"Checkpoint loaded from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")

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
    embedding_dim = 80  # Modified embedding dimension
    
    model = ANCFModel(num_users, num_items, embedding_dim)
    
    # Load checkpoint if available
    checkpoint_path = 'best_model.npy'
    load_checkpoint(model, checkpoint_path)
    
    # Define training parameters
    epochs = 20  # Increased epochs
    initial_lr = 0.01  # Start with higher learning rate, can be changed to experiment
    
    # Train the model with early stopping and learning rate scheduling
    train_model(model, train_batches, val_batches, epochs, initial_lr, checkpoint_path=checkpoint_path)

if __name__ == "__main__":
    main()
'''

from ancf_models import ANCFModel
from preprocess import preprocess_data
from dataloader import batch_data
import numpy as np
import matplotlib.pyplot as plt
import os

def lr_schedule(epoch, lr):
    if epoch > 10:
        return lr * 0.5
    return lr

def train_model(model, train_batches, val_batches, epochs, initial_lr, patience=3, decay_factor=0.9, checkpoint_path='best_model.npy'):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    learning_rate = initial_lr
    
    for epoch in range(epochs):
        train_loss = 0
        
        for X_batch, y_batch in train_batches:
            user_ids = X_batch['user_id'].values
            item_ids = X_batch['product_id'].values
            
            # Forward pass
            predictions = model.forward(user_ids, item_ids)
            loss = np.mean((predictions - y_batch) ** 2)  # MSE loss
            train_loss += loss
            
            # Backward pass
            model.learning_rate = learning_rate  # Apply the dynamic learning rate
            model.backward(user_ids, item_ids, predictions, y_batch)
        
        avg_train_loss = train_loss / len(train_batches)
        train_losses.append(avg_train_loss)
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}')
        
        # Validate on validation set
        val_loss = validate_model(model, val_batches)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}')
        
        # Save model checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, checkpoint_path)
            print(f"Model checkpoint saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
        
        # Apply learning rate decay after each epoch
        learning_rate *= decay_factor
        print(f"Epoch {epoch+1}, Learning Rate: {learning_rate}")

        # Debug statements
        print("Epoch:", epoch + 1)
        print("User Embeddings:", model.user_embedding)
        print("Item Embeddings:", model.item_embedding)

        # Plot training and validation loss
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.savefig('training_validation_loss.png')

def validate_model(model, val_batches):
    val_loss = 0
    for X_batch, y_batch in val_batches:
        user_ids = X_batch['user_id'].values
        item_ids = X_batch['product_id'].values
        predictions = model.forward(user_ids, item_ids)
        loss = np.mean((predictions - y_batch) ** 2)  # MSE loss
        val_loss += loss
    return val_loss / len(val_batches)

def save_checkpoint(model, checkpoint_path):
    """Save the model state to a file."""
    np.save(checkpoint_path, {
        'user_embedding': model.user_embedding,
        'item_embedding': model.item_embedding,
        'weights': model.weights,
        'bias': model.bias
    })

def load_checkpoint(model, checkpoint_path):
    """Load the model state from a file."""
    if os.path.exists(checkpoint_path):
        checkpoint = np.load(checkpoint_path, allow_pickle=True).item()
        model.user_embedding = checkpoint['user_embedding']
        model.item_embedding = checkpoint['item_embedding']
        model.weights = checkpoint['weights']
        model.bias = checkpoint['bias']
        print(f"Checkpoint loaded from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")

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
    embedding_dim = 80  # Modified embedding dimension
    
    model = ANCFModel(num_users, num_items, embedding_dim)
    
    # Load checkpoint if available
    checkpoint_path = 'best_model.npy'
    load_checkpoint(model, checkpoint_path)
    
    # Define training parameters
    epochs = 20  # Increased epochs
    initial_lr = 0.01  # Start with higher learning rate, can be changed to experiment
    
    # Train the model with early stopping and learning rate scheduling
    train_model(model, train_batches, val_batches, epochs, initial_lr, checkpoint_path=checkpoint_path)

if __name__ == "__main__":
    main()
