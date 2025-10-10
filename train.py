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
        
        # Training step
        for X_batch, y_batch in train_batches:
            user_ids = X_batch[:, 0]  # Adjust to match shape (user_id, item_id)
            item_ids = X_batch[:, 1]
            
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
        
        # Validation step
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

        # Plot training and validation loss
        plot_losses(train_losses, val_losses)

def validate_model(model, val_batches):
    val_loss = 0
    for X_batch, y_batch in val_batches:
        user_ids = X_batch[:, 0]
        item_ids = X_batch[:, 1]
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

def plot_losses(train_losses, val_losses):
    """Plot training and validation losses."""
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.savefig('training_validation_loss.png')

def main():
    # Data preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test, _, _ = preprocess_data()
    
    # Convert data to NumPy arrays if they are not already
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    
    # Data batching
    batch_size = 64
    train_batches = list(batch_data(X_train, y_train, batch_size))
    val_batches = list(batch_data(X_val, y_val, batch_size))
    
    # Instantiate the model
    num_users = max(X_train[:, 0]) + 1  # Adjust to match shape (user_id, item_id)
    num_items = max(X_train[:, 1]) + 1
    embedding_dim = 80  # Modified embedding dimension
    
    model = ANCFModel(num_users, num_items, embedding_dim)
    
    # Load checkpoint if available
    checkpoint_path = 'best_model.npy'
    load_checkpoint(model, checkpoint_path)
    
    # Define training parameters
    epochs = 20  # Increased epochs
    initial_lr = 0.001  # Start with higher learning rate
    
    # Train the model with early stopping and learning rate scheduling
    train_model(model, train_batches, val_batches, epochs, initial_lr, checkpoint_path=checkpoint_path)

if __name__ == "__main__":
    main()


# train.py

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
        
        # Training step
        for X_batch, y_batch in train_batches:
            user_ids = X_batch[:, 0]  # Adjust to match shape (user_id, item_id)
            item_ids = X_batch[:, 1]
            
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
        
        # Validation step
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

        # Plot training and validation loss
        plot_losses(train_losses, val_losses)

def validate_model(model, val_batches):
    val_loss = 0
    for X_batch, y_batch in val_batches:
        user_ids = X_batch[:, 0]
        item_ids = X_batch[:, 1]
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

def plot_losses(train_losses, val_losses):
    """Plot training and validation losses."""
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.savefig('training_validation_loss.png')

def main():
    # Data preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test, _, _ = preprocess_data()
    
    # Convert data to NumPy arrays if they are not already
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    
    # Data batching
    batch_size = 64
    train_batches = list(batch_data(X_train, y_train, batch_size))
    val_batches = list(batch_data(X_val, y_val, batch_size))
    
    # Instantiate the model
    num_users = max(X_train[:, 0]) + 1  # Adjust to match shape (user_id, item_id)
    num_items = max(X_train[:, 1]) + 1
    embedding_dim = 80  # Modified embedding dimension
    
    model = ANCFModel(num_users, num_items, embedding_dim)
    
    # Load checkpoint if available
    checkpoint_path = 'ancf_model.npz'
    load_checkpoint(model, checkpoint_path)
    
    # Define training parameters
    epochs = 20  # Increased epochs
    initial_lr = 0.001  # Start with higher learning rate
    
    # Train the model with early stopping and learning rate scheduling
    train_model(model, train_batches, val_batches, epochs, initial_lr, checkpoint_path=checkpoint_path)

if __name__ == "__main__":
    main()


import numpy as np
import matplotlib.pyplot as plt

def train_model(model, train_batches, val_batches, epochs, lr):
    print("Starting training...", flush=True)
    print(f"Epochs = {epochs}, Train batches = {len(train_batches)}, Val batches = {len(val_batches)}", flush=True)

    best_val_loss = float('inf')
    patience = 2
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_loss = 0
        for X_batch, y_batch in train_batches:
            user_ids, item_ids = X_batch[:, 0], X_batch[:, 1]
            preds = model.predict(user_ids, item_ids)
            loss = np.mean((preds - y_batch) ** 2)

            # simple SGD
            grad = (preds - y_batch).mean()
            model.W_out -= lr * grad * model.W_out
            train_loss += loss

        if len(train_batches) == 0:
            print("⚠️ No training batches found.")
            break

        train_loss /= len(train_batches)

        # Validation
        val_loss = 0
        for X_batch, y_batch in val_batches:
            preds = model.predict(X_batch[:, 0], X_batch[:, 1])
            val_loss += np.mean((preds - y_batch) ** 2)
        val_loss /= len(val_batches)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", flush=True)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            np.save('best_model.npy', model.__dict__)
            print("✅ Saved best model.", flush=True)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⛔ Early stopping.")
                break

        lr *= 0.9  # decay learning rate

    # === Plot losses and save ===
    plt.plot(train_losses, label="Training Loss", color='royalblue')
    plt.plot(val_losses, label="Validation Loss", color='tomato')
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    print("Running preprocessing pipeline...", flush=True)
    from ancf_models import ANCFModel
    from dataloader import batch_data

    # 1️⃣ Preprocess data
    from preprocess import preprocess_data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()

    # 2️⃣ Create batches
    train_batches = list(batch_data(X_train, y_train, batch_size=64))
    val_batches = list(batch_data(X_val, y_val, batch_size=64))

    # 3️⃣ Initialize model
    # Use full dataset to determine embedding sizes
    num_users = max(X_train["user_id"].max(), X_val["user_id"].max(), X_test["user_id"].max()) + 1
    num_items = max(X_train["product_id"].max(), X_val["product_id"].max(), X_test["product_id"].max()) + 1

    model = ANCFModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=64
    )

    # 4️⃣ Train model
    from train import train_model
    train_model(model, train_batches, val_batches, epochs=25, lr=0.01)

    print("✅ Training finished. Loss curve saved to loss_curve.png")
'''

import numpy as np
import matplotlib.pyplot as plt

def train_model_attention(model, train_batches, val_batches, user_histories_dict, epochs=30, lr=0.01):
    """
    train_batches: list of (X_batch, y_batch), X_batch[:,0]=user_ids, X_batch[:,1]=item_ids
    user_histories_dict: dict {user_id: list of item_ids user interacted with}
    """
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        train_loss = 0
        for X_batch, y_batch in train_batches:
            # --- Ensure NumPy arrays ---
            if not isinstance(X_batch, np.ndarray):
                X_batch = X_batch.to_numpy()
            if not isinstance(y_batch, np.ndarray):
                y_batch = y_batch.to_numpy()

            user_ids, item_ids = X_batch[:,0], X_batch[:,1]
            batch_size = len(y_batch)

            X_concat = []
            contexts = []

            for u, i in zip(user_ids, item_ids):
                user_emb = model.user_embeddings[u]
                hist = user_histories_dict.get(u, [])
                hist_embs = model.item_embeddings[hist] if len(hist) > 0 else np.zeros((1, model.embedding_dim))

                # attention
                Q = user_emb @ model.Wq
                K = hist_embs @ model.Wk
                V = hist_embs @ model.Wv
                attn_scores = (Q @ K.T) / np.sqrt(model.embedding_dim)
                attn_weights = np.exp(attn_scores - attn_scores.max())
                attn_weights /= attn_weights.sum()
                context = attn_weights @ V

                x_vec = np.concatenate([user_emb, context])
                X_concat.append(x_vec)
                contexts.append((context, attn_weights, hist))

            X_concat = np.array(X_concat)
            preds = X_concat @ model.W_out + model.b_out
            errors = (preds.flatten() - y_batch)  # NumPy array

            loss = np.mean(errors**2)
            train_loss += loss

            # --- Gradients ---
            grad_W_out = (2 / batch_size) * X_concat.T @ errors[:, None]
            grad_b_out = (2 / batch_size) * errors.sum()

            model.W_out -= lr * grad_W_out
            model.b_out -= lr * grad_b_out

            # Update embeddings
            for idx, (u, i) in enumerate(zip(user_ids, item_ids)):
                error = errors[idx]
                grad_user = (2 / batch_size) * model.W_out[:model.embedding_dim].flatten() * error
                model.user_embeddings[u] -= lr * grad_user

                grad_item = (2 / batch_size) * model.W_out[model.embedding_dim:].flatten() * error
                model.item_embeddings[i] -= lr * grad_item

                # Historical items
                context_vec, attn_weights, hist = contexts[idx]
                for h_idx, attn_w in zip(hist, attn_weights):
                    grad_hist_item = (2 / batch_size) * model.W_out[model.embedding_dim:].flatten() * error * attn_w
                    model.item_embeddings[h_idx] -= lr * grad_hist_item

        train_loss /= len(train_batches)
        train_losses.append(train_loss)

        # --- Validation ---
        val_loss = 0
        for X_batch, y_batch in val_batches:
            if not isinstance(X_batch, np.ndarray):
                X_batch = X_batch.to_numpy()
            if not isinstance(y_batch, np.ndarray):
                y_batch = y_batch.to_numpy()

            user_ids, item_ids = X_batch[:,0], X_batch[:,1]

            preds_val = []
            for u, i in zip(user_ids, item_ids):
                user_emb = model.user_embeddings[u]
                hist = user_histories_dict.get(u, [])
                hist_embs = model.item_embeddings[hist] if len(hist) > 0 else np.zeros((1, model.embedding_dim))

                Q = user_emb @ model.Wq
                K = hist_embs @ model.Wk
                V = hist_embs @ model.Wv
                attn_scores = (Q @ K.T) / np.sqrt(model.embedding_dim)
                attn_weights = np.exp(attn_scores - attn_scores.max())
                attn_weights /= attn_weights.sum()
                context = attn_weights @ V
                x_vec = np.concatenate([user_emb, context])
                y_pred = x_vec @ model.W_out + model.b_out
                preds_val.append(y_pred)

            preds_val = np.array(preds_val).flatten()
            val_loss += np.mean((preds_val - y_batch)**2)

        val_loss /= len(val_batches)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            np.save('best_model_attention.npy', model.__dict__)
            print("✅ Saved best model.")

    return train_losses, val_losses

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ancf_models import ANCFModelAttention
    from dataloader import batch_data
    from preprocess import preprocess_data

    print("Running preprocessing pipeline...", flush=True)
    X_train, X_val, X_test, y_train, y_val, y_test, num_users, num_items = preprocess_data()

    # Create batches
    train_batches = list(batch_data(X_train, y_train, batch_size=64))
    val_batches = list(batch_data(X_val, y_val, batch_size=64))

    # Initialize model
    num_users = max(X_train["user_id"].max(), X_val["user_id"].max(), X_test["user_id"].max()) + 1
    num_items = max(X_train["product_id"].max(), X_val["product_id"].max(), X_test["product_id"].max()) + 1

    model = ANCFModelAttention(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=4
    )

    # Build user_histories dictionary from training set
    user_histories = {}
    for u, i in zip(X_train["user_id"], X_train["product_id"]):
        if u not in user_histories:
            user_histories[u] = []
        user_histories[u].append(i)

    # Train the model
    train_losses, val_losses = train_model_attention(
        model, train_batches, val_batches, user_histories, epochs=30, lr=0.01
    )

    # Plot losses
    plt.plot(train_losses, label="Training Loss", color='royalblue')
    plt.plot(val_losses, label="Validation Loss", color='tomato')
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=300)
    plt.show()

    print("✅ Training finished. Loss curve saved to loss_curve.png")
