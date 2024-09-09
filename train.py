'''
from ancf_models import ANCFModel
from preprocess import preprocess_data
from dataloader import batch_data
import numpy as np
import matplotlib.pyplot as plt

def train_model(model, train_batches, val_batches, epochs, initial_lr, patience=3, decay_factor=0.9):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    learning_rate = initial_lr
    
    for epoch in range(epochs):
        #model.train_mode()  # Switch to training mode
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
        
        # Early Stopping: Stop if validation loss does not improve
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
        
        # Apply learning rate decay after each epoch
        learning_rate *= decay_factor
        print(f"Epoch {epoch+1}, Learning Rate: {learning_rate}")
    
    # Plot training and validation loss
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.savefig('training_validation_loss.png')

def validate_model(model, val_batches):
    #model.eval_mode()  # Switch to evaluation mode
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
    embedding_dim = 80  # Modified embedding dimension
    
    model = ANCFModel(num_users, num_items, embedding_dim)
    
    # Define training parameters
    epochs = 20  # Increased epochs
    initial_lr = 0.01  # Start with higher learning rate, can be changed to experiment
    
    # Train the model with early stopping and learning rate scheduling
    train_model(model, train_batches, val_batches, epochs, initial_lr)
    
    # Evaluate the model (optional)
    # from training.eval import evaluate_model
    # ndcg, precision, recall = evaluate_model(model, test_batches)
    # print(f'NDCG: {ndcg}, Precision: {precision}, Recall: {recall}')

if __name__ == "__main__":
    main()

from ancf_models import ANCFModel
from preprocess import preprocess_data
from dataloader import batch_data
import numpy as np
import matplotlib.pyplot as plt

def train_model(model, train_batches, val_batches, epochs, initial_lr, patience=3, decay_factor=0.9):
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

        # Early Stopping: Stop if validation loss does not improve
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Apply learning rate decay after each epoch
        learning_rate *= decay_factor
        print(f"Epoch {epoch+1}, Learning Rate: {learning_rate}")

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

    # Define training parameters
    epochs = 20  # Increased epochs
    initial_lr = 0.01  # Start with higher learning rate, can be changed to experiment

    # Train the model with early stopping and learning rate scheduling
    train_model(model, train_batches, val_batches, epochs, initial_lr)

if __name__ == "__main__":
    main()


import tensorflow as tf
from ancf_models import ANCFModel
from preprocess import preprocess_data
from dataloader import batch_data
import numpy as np
import matplotlib.pyplot as plt

def train_model(model, train_batches, val_batches, epochs, initial_lr, patience=3, decay_factor=0.9):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    learning_rate = initial_lr
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)

    for epoch in range(epochs):
        train_loss = 0

        for X_batch, y_batch in train_batches:
            user_ids = X_batch['user_id'].values
            item_ids = X_batch['product_id'].values

            # Convert to TensorFlow tensors
            user_ids = tf.convert_to_tensor(user_ids, dtype=tf.int32)
            item_ids = tf.convert_to_tensor(item_ids, dtype=tf.int32)
            y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)

            with tf.GradientTape() as tape:
                # Forward pass
                predictions = model(user_ids, item_ids)
                loss = model.compute_loss(predictions, y_batch)

            # Backward pass
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss += loss.numpy()

        avg_train_loss = train_loss / len(train_batches)
        train_losses.append(avg_train_loss)
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}')

        # Validate on validation set
        val_loss = validate_model(model, val_batches)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}')

        # Early Stopping: Stop if validation loss does not improve
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Apply learning rate decay after each epoch
        learning_rate *= decay_factor
        optimizer.learning_rate.assign(learning_rate)
        print(f"Epoch {epoch+1}, Learning Rate: {learning_rate}")

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

        # Convert to TensorFlow tensors
        user_ids = tf.convert_to_tensor(user_ids, dtype=tf.int32)
        item_ids = tf.convert_to_tensor(item_ids, dtype=tf.int32)
        y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)

        predictions = model(user_ids, item_ids)
        loss = model.compute_loss(predictions, y_batch)
        val_loss += loss.numpy()
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
    embedding_dim = 80  # Modified embedding dimension

    model = ANCFModel(num_users, num_items, embedding_dim)

    # Define training parameters
    epochs = 20  # Increased epochs
    initial_lr = 0.01  # Start with higher learning rate

    # Train the model with early stopping and learning rate scheduling
    train_model(model, train_batches, val_batches, epochs, initial_lr)

if __name__ == "__main__":
    main()
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, ndcg_score
import os

def train_model(model, train_batches, val_batches, epochs, initial_lr, patience=3, decay_factor=0.9, save_dir='./saved_models'):
    """
    Train the ANCF model with early stopping, learning rate scheduling, and model saving.

    Args:
        model (tf.keras.Model): The ANCF model to train.
        train_batches (list): List of training data batches.
        val_batches (list): List of validation data batches.
        epochs (int): Number of training epochs.
        initial_lr (float): Initial learning rate.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        decay_factor (float): Factor by which the learning rate will be reduced each epoch.
        save_dir (str): Directory to save model checkpoints.

    Returns:
        dict: Dictionary containing training history (losses and metrics).
    """
    train_losses, val_losses = [], []
    train_ndcgs, val_ndcgs = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    learning_rate = initial_lr
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(epochs):
        train_loss, train_ndcg = 0, 0
        for X_batch, y_batch in train_batches:
            user_ids = tf.convert_to_tensor(X_batch['user_id'].values, dtype=tf.int32)
            item_ids = tf.convert_to_tensor(X_batch['product_id'].values, dtype=tf.int32)
            y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)

            with tf.GradientTape() as tape:
                predictions = model(user_ids, item_ids)
                loss = model.compute_loss(predictions, y_batch)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss += loss.numpy()
            train_ndcg += ndcg_score([y_batch.numpy()], [predictions.numpy()])

        avg_train_loss = train_loss / len(train_batches)
        avg_train_ndcg = train_ndcg / len(train_batches)
        train_losses.append(avg_train_loss)
        train_ndcgs.append(avg_train_ndcg)

        val_loss, val_ndcg = validate_model(model, val_batches)
        val_losses.append(val_loss)
        val_ndcgs.append(val_ndcg)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train NDCG: {avg_train_ndcg:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val NDCG: {val_ndcg:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_weights(os.path.join(save_dir, 'best_model.h5'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        learning_rate *= decay_factor
        optimizer.learning_rate.assign(learning_rate)

    plot_training_history(train_losses, val_losses, train_ndcgs, val_ndcgs)
    return {'train_loss': train_losses, 'val_loss': val_losses, 
            'train_ndcg': train_ndcgs, 'val_ndcg': val_ndcgs}

def validate_model(model, val_batches):
    val_loss, val_ndcg = 0, 0
    for X_batch, y_batch in val_batches:
        user_ids = tf.convert_to_tensor(X_batch['user_id'].values, dtype=tf.int32)
        item_ids = tf.convert_to_tensor(X_batch['product_id'].values, dtype=tf.int32)
        y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)

        predictions = model(user_ids, item_ids)
        loss = model.compute_loss(predictions, y_batch)
        val_loss += loss.numpy()
        val_ndcg += ndcg_score([y_batch.numpy()], [predictions.numpy()])

    return val_loss / len(val_batches), val_ndcg / len(val_batches)

def plot_training_history(train_losses, val_losses, train_ndcgs, val_ndcgs):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_ndcgs, label='Training NDCG')
    plt.plot(val_ndcgs, label='Validation NDCG')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('NDCG')
    plt.title('Training and Validation NDCG')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()