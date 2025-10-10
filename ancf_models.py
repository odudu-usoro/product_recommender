'''
import numpy as np
import os

class ANCFModel:
    def __init__(self, num_users, num_items, embedding_dim, learning_rate=0.01, l2_lambda=0.001):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        
        # Initialize embeddings with the correct embedding dimension
        self.user_embedding = np.random.normal(scale=1./self.embedding_dim, size=(num_users, embedding_dim))
        self.item_embedding = np.random.normal(scale=1./self.embedding_dim, size=(num_items, embedding_dim))
        
        # Initialize weights with the correct embedding dimension
        self.Wq = np.random.normal(scale=1./embedding_dim, size=(embedding_dim, embedding_dim))
        self.Wk = np.random.normal(scale=1./embedding_dim, size=(embedding_dim, embedding_dim))
        self.Wv = np.random.normal(scale=1./embedding_dim, size=(embedding_dim, embedding_dim))
        self.weights = np.random.normal(scale=1./embedding_dim, size=(embedding_dim, 1))
        self.bias = 0.0

    def apply_regularization(self):
        self.weights -= self.l2_lambda * self.weights
        self.user_embedding -= self.l2_lambda * self.user_embedding
        self.item_embedding -= self.l2_lambda * self.item_embedding

    def self_attention(self, user_emb, item_emb):
        # Compute Query, Key, and Value vectors
        Q = np.dot(user_emb, self.Wq)  # Shape: (batch_size, embedding_dim)
        K = np.dot(item_emb, self.Wk)  # Shape: (batch_size, embedding_dim)
        V = np.dot(item_emb, self.Wv)  # Shape: (batch_size, embedding_dim)
        
        # Compute attention scores: Q * K^T / sqrt(d)
        attention_scores = np.dot(Q, K.T) / np.sqrt(self.embedding_dim)  # Shape: (batch_size, batch_size)
        
        # Apply softmax to normalize attention scores
        attention_weights = np.exp(attention_scores - np.max(attention_scores, axis=1, keepdims=True))  # Numerical stability
        attention_weights /= np.sum(attention_weights, axis=1, keepdims=True)
        
        # Compute weighted sum of value vectors
        attention_output = np.dot(attention_weights, V)  # Shape: (batch_size, embedding_dim)
        
        return attention_output

    def forward(self, user_ids, item_ids):
        # Retrieve embeddings
        user_emb = self.user_embedding[user_ids]  # Shape: (batch_size, embedding_dim)
        item_emb = self.item_embedding[item_ids]  # Shape: (batch_size, embedding_dim)
        
        # Apply self-attention mechanism
        attention_output = self.self_attention(user_emb, item_emb)
        
        # Final linear prediction
        predictions = np.dot(attention_output, self.weights) + self.bias  # Shape: (batch_size, 1)
        
        # Apply ReLU activation for non-linearity and ensure predictions are within range [1, 5]
        predictions = np.maximum(1, predictions)  # Ensure minimum value of 1
        predictions = np.minimum(5, predictions)  # Ensure maximum value of 5
        
        return predictions.flatten()  # Shape: (batch_size,)
    
    def predict(self, user_ids, item_ids):
        return self.forward(user_ids, item_ids) 

    def backward(self, user_ids, item_ids, predictions, y_true):
        # Compute gradients for MSE loss
        error = predictions - y_true  # Ensure y_true is a NumPy array

        # Split the attention_output back into user and item parts
        attention_output = self.self_attention(self.user_embedding[user_ids], self.item_embedding[item_ids])
        
        # Gradients for weights and bias using only the attention_output (not concatenated)
        grad_weights = np.dot(attention_output.T, error) / len(user_ids)  # Shape: (embedding_dim, 1)
        grad_bias = np.mean(error)
        
        # Update weights and bias
        self.weights -= self.learning_rate * grad_weights.reshape(self.weights.shape)  # This should now work
        self.bias -= self.learning_rate * grad_bias

        # Apply L2 regularization
        self.apply_regularization()

        # Compute gradients for embeddings
        grad_combined = np.dot(error.reshape(-1, 1), self.weights.T)  # Shape: (batch_size, embedding_dim)

        # Update embeddings
        self.user_embedding[user_ids] -= self.learning_rate * grad_combined
        self.item_embedding[item_ids] -= self.learning_rate * grad_combined

    def train_on_batch(self, batch_user_ids, batch_item_ids, batch_ratings):
        predictions = self.forward(batch_user_ids, batch_item_ids)
        self.backward(batch_user_ids, batch_item_ids, predictions, batch_ratings)

    def load_checkpoint(self, checkpoint_path):
    #Load the model state from a file
        if os.path.exists(checkpoint_path):
            checkpoint = np.load(checkpoint_path, allow_pickle=True).item()
            self.user_embedding = checkpoint['user_embedding']
            self.item_embedding = checkpoint['item_embedding']
            self.weights = checkpoint['weights']
            self.bias = checkpoint['bias']
            print(f"Checkpoint loaded from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")


import numpy as np
import os

class ANCFModel:
    def __init__(self, num_users, num_items, embedding_dim, learning_rate=0.01, l2_lambda=0.001):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        
        # Initialize embeddings with the correct embedding dimension
        self.user_embedding = np.random.normal(scale=1./self.embedding_dim, size=(num_users, embedding_dim))
        self.item_embedding = np.random.normal(scale=1./self.embedding_dim, size=(num_items, embedding_dim))
        
        # Initialize weights with the correct embedding dimension
        self.Wq = np.random.normal(scale=1./embedding_dim, size=(embedding_dim, embedding_dim))
        self.Wk = np.random.normal(scale=1./embedding_dim, size=(embedding_dim, embedding_dim))
        self.Wv = np.random.normal(scale=1./embedding_dim, size=(embedding_dim, embedding_dim))
        self.weights = np.random.normal(scale=1./embedding_dim, size=(embedding_dim, 1))
        self.bias = 0.0

    def apply_regularization(self):
        self.weights -= self.l2_lambda * self.weights
        self.user_embedding -= self.l2_lambda * self.user_embedding
        self.item_embedding -= self.l2_lambda * self.item_embedding

    def self_attention(self, user_emb, item_emb):
        # Compute Query, Key, and Value vectors
        Q = np.dot(user_emb, self.Wq)  # Shape: (batch_size, embedding_dim)
        K = np.dot(item_emb, self.Wk)  # Shape: (batch_size, embedding_dim)
        V = np.dot(item_emb, self.Wv)  # Shape: (batch_size, embedding_dim)
        
        # Compute attention scores: Q * K^T / sqrt(d)
        attention_scores = np.dot(Q, K.T) / np.sqrt(self.embedding_dim)  # Shape: (batch_size, batch_size)
        
        # Apply softmax to normalize attention scores
        attention_weights = np.exp(attention_scores - np.max(attention_scores, axis=1, keepdims=True))  # Numerical stability
        attention_weights /= np.sum(attention_weights, axis=1, keepdims=True)
        
        # Compute weighted sum of value vectors
        attention_output = np.dot(attention_weights, V)  # Shape: (batch_size, embedding_dim)
        
        return attention_output

    def forward(self, user_ids, item_ids):
        # Ensure user_ids and item_ids are within bounds
        user_ids = np.clip(user_ids, 0, self.user_embedding.shape[0] - 1)
        item_ids = np.clip(item_ids, 0, self.item_embedding.shape[0] - 1)
        
        # Retrieve embeddings
        user_emb = self.user_embedding[user_ids]  # Shape: (batch_size, embedding_dim)
        item_emb = self.item_embedding[item_ids]  # Shape: (batch_size, embedding_dim)
        
        # Apply self-attention mechanism
        attention_output = self.self_attention(user_emb, item_emb)
        
        # Final linear prediction
        predictions = np.dot(attention_output, self.weights) + self.bias  # Shape: (batch_size, 1)
        
        # Apply ReLU activation for non-linearity and ensure predictions are within range [1, 5]
        predictions = np.maximum(1, predictions)  # Ensure minimum value of 1
        predictions = np.minimum(5, predictions)  # Ensure maximum value of 5
        
        return predictions.flatten()  # Shape: (batch_size,)
    
    def predict(self, user_ids, item_ids):
        return self.forward(user_ids, item_ids) 

    def backward(self, user_ids, item_ids, predictions, y_true):
        # Compute gradients for MSE loss
        error = predictions - y_true  # Ensure y_true is a NumPy array

        # Split the attention_output back into user and item parts
        attention_output = self.self_attention(self.user_embedding[user_ids], self.item_embedding[item_ids])
        
        # Gradients for weights and bias using only the attention_output (not concatenated)
        grad_weights = np.dot(attention_output.T, error) / len(user_ids)  # Shape: (embedding_dim, 1)
        grad_bias = np.mean(error)
        
        # Update weights and bias
        self.weights -= self.learning_rate * grad_weights.reshape(self.weights.shape)  # This should now work
        self.bias -= self.learning_rate * grad_bias

        # Apply L2 regularization
        self.apply_regularization()

        # Compute gradients for embeddings
        grad_combined = np.dot(error.reshape(-1, 1), self.weights.T)  # Shape: (batch_size, embedding_dim)

        # Update embeddings
        self.user_embedding[user_ids] -= self.learning_rate * grad_combined
        self.item_embedding[item_ids] -= self.learning_rate * grad_combined

    def train_on_batch(self, batch_user_ids, batch_item_ids, batch_ratings):
        predictions = self.forward(batch_user_ids, batch_item_ids)
        self.backward(batch_user_ids, batch_item_ids, predictions, batch_ratings)

    def load_checkpoint(self, checkpoint_path):
        # Load the model state from a file
        if os.path.exists(checkpoint_path):
            checkpoint = np.load(checkpoint_path, allow_pickle=True).item()
            self.user_embedding = checkpoint['user_embedding']
            self.item_embedding = checkpoint['item_embedding']
            self.weights = checkpoint['weights']
            self.bias = checkpoint['bias']
            print(f"Checkpoint loaded from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")


import numpy as np
import os

class ANCFModel:
    def __init__(self, num_users, num_items, embedding_dim, learning_rate=0.01, l2_lambda=0.001):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        
        # Initialize embeddings with the correct embedding dimension
        self.user_embedding = np.random.normal(scale=1./self.embedding_dim, size=(num_users, embedding_dim))
        self.item_embedding = np.random.normal(scale=1./self.embedding_dim, size=(num_items, embedding_dim))
        
        # Initialize weights with the correct embedding dimension
        self.Wq = np.random.normal(scale=1./embedding_dim, size=(embedding_dim, embedding_dim))
        self.Wk = np.random.normal(scale=1./embedding_dim, size=(embedding_dim, embedding_dim))
        self.Wv = np.random.normal(scale=1./embedding_dim, size=(embedding_dim, embedding_dim))
        self.weights = np.random.normal(scale=1./embedding_dim, size=(embedding_dim, 1))
        self.bias = 0.0

    def apply_regularization(self):
        self.weights -= self.l2_lambda * self.weights
        self.user_embedding -= self.l2_lambda * self.user_embedding
        self.item_embedding -= self.l2_lambda * self.item_embedding

    def self_attention(self, user_emb, item_emb):
        # Compute Query, Key, and Value vectors
        Q = np.dot(user_emb, self.Wq)  # Shape: (batch_size, embedding_dim)
        K = np.dot(item_emb, self.Wk)  # Shape: (batch_size, embedding_dim)
        V = np.dot(item_emb, self.Wv)  # Shape: (batch_size, embedding_dim)
        
        # Compute attention scores: Q * K^T / sqrt(d)
        attention_scores = np.dot(Q, K.T) / np.sqrt(self.embedding_dim)  # Shape: (batch_size, batch_size)
        
        # Apply softmax to normalize attention scores
        attention_weights = np.exp(attention_scores - np.max(attention_scores, axis=1, keepdims=True))  # Numerical stability
        attention_weights /= np.sum(attention_weights, axis=1, keepdims=True)
        
        # Compute weighted sum of value vectors
        attention_output = np.dot(attention_weights, V)  # Shape: (batch_size, embedding_dim)
        
        return attention_output

    def forward(self, user_ids, item_ids):
        # Ensure user_ids and item_ids are within bounds
        user_ids = np.clip(user_ids, 0, self.user_embedding.shape[0] - 1)
        item_ids = np.clip(item_ids, 0, self.item_embedding.shape[0] - 1)
        
        # Retrieve embeddings
        user_emb = self.user_embedding[user_ids]  # Shape: (batch_size, embedding_dim)
        item_emb = self.item_embedding[item_ids]  # Shape: (batch_size, embedding_dim)
        
        # Apply self-attention mechanism
        attention_output = self.self_attention(user_emb, item_emb)
        
        # Final linear prediction
        predictions = np.dot(attention_output, self.weights) + self.bias  # Shape: (batch_size, 1)
        
        # Apply ReLU activation for non-linearity and ensure predictions are within range [1, 5]
        predictions = np.maximum(1, predictions)  # Ensure minimum value of 1
        predictions = np.minimum(5, predictions)  # Ensure maximum value of 5
        
        return predictions.flatten()  # Shape: (batch_size,)
    
    def predict(self, user_ids, item_ids):
        return self.forward(user_ids, item_ids) 

    def backward(self, user_ids, item_ids, predictions, y_true):
        # Compute gradients for MSE loss
        error = predictions - y_true  # Ensure y_true is a NumPy array

        # Split the attention_output back into user and item parts
        attention_output = self.self_attention(self.user_embedding[user_ids], self.item_embedding[item_ids])
        
        # Gradients for weights and bias using only the attention_output (not concatenated)
        grad_weights = np.dot(attention_output.T, error) / len(user_ids)  # Shape: (embedding_dim, 1)
        grad_bias = np.mean(error)
        
        # Update weights and bias
        self.weights -= self.learning_rate * grad_weights.reshape(self.weights.shape)  # This should now work
        self.bias -= self.learning_rate * grad_bias

        # Apply L2 regularization
        self.apply_regularization()

        # Compute gradients for embeddings
        grad_combined = np.dot(error.reshape(-1, 1), self.weights.T)  # Shape: (batch_size, embedding_dim)

        # Update embeddings
        self.user_embedding[user_ids] -= self.learning_rate * grad_combined
        self.item_embedding[item_ids] -= self.learning_rate * grad_combined

    def train_on_batch(self, batch_user_ids, batch_item_ids, batch_ratings):
        predictions = self.forward(batch_user_ids, batch_item_ids)
        self.backward(batch_user_ids, batch_item_ids, predictions, batch_ratings)

    def save_checkpoint(self, checkpoint_path):
        # Create a dictionary to hold the model's state
        checkpoint = {
            'user_embedding': self.user_embedding,
            'item_embedding': self.item_embedding,
            'weights': self.weights,
            'bias': self.bias
        }
        
        # Save the dictionary to a .npz file
        np.savez(checkpoint_path, **checkpoint)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        # Load the model state from a file
        if os.path.exists(checkpoint_path):
            checkpoint = np.load(checkpoint_path, allow_pickle=True).item()
            self.user_embedding = checkpoint['user_embedding']
            self.item_embedding = checkpoint['item_embedding']
            self.weights = checkpoint['weights']
            self.bias = checkpoint['bias']
            print(f"Checkpoint loaded from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")


import numpy as np

class ANCFModel:
    def __init__(self, num_users, num_items, embedding_dim=32, dropout_rate=0.3):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate

        self.user_embeddings = np.random.normal(0, 0.1, (num_users, embedding_dim))
        self.item_embeddings = np.random.normal(0, 0.1, (num_items, embedding_dim))

        # Weights for attention
        self.Wq = np.random.normal(0, 0.1, (embedding_dim, embedding_dim))
        self.Wk = np.random.normal(0, 0.1, (embedding_dim, embedding_dim))
        self.Wv = np.random.normal(0, 0.1, (embedding_dim, embedding_dim))

        # Prediction layer
        self.W_out = np.random.normal(0, 0.1, (embedding_dim * 2, 1))
        self.b_out = np.zeros((1,))

    def attention(self, user_emb, item_emb):
        Q = user_emb @ self.Wq
        K = item_emb @ self.Wk
        V = item_emb @ self.Wv
        attn_scores = np.dot(Q, K) / np.sqrt(self.embedding_dim)
        attn_weights = np.exp(attn_scores) / np.sum(np.exp(attn_scores))
        context = attn_weights * V
        return context

    def predict(self, user_ids, item_ids):
        preds = []
        for u, i in zip(user_ids, item_ids):
            user_emb = self.user_embeddings[u]
            item_emb = self.item_embeddings[i]
            context = self.attention(user_emb, item_emb)
            x = np.concatenate([user_emb, context])
            y_pred = x @ self.W_out + self.b_out
            preds.append(y_pred)
        return np.array(preds).flatten()
'''

import numpy as np

class ANCFModelAttention:
    def __init__(self, num_users, num_items, embedding_dim=4, dropout_rate=0.0):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate

        # Embeddings
        self.user_embeddings = np.random.normal(0, 0.1, (num_users, embedding_dim))
        self.item_embeddings = np.random.normal(0, 0.1, (num_items, embedding_dim))

        # Attention weights
        self.Wq = np.random.normal(0, 0.1, (embedding_dim, embedding_dim))
        self.Wk = np.random.normal(0, 0.1, (embedding_dim, embedding_dim))
        self.Wv = np.random.normal(0, 0.1, (embedding_dim, embedding_dim))

        # Output layer
        self.W_out = np.random.normal(0, 0.1, (embedding_dim * 2, 1))
        self.b_out = np.zeros((1,))

    def attention(self, user_emb, item_hist_embs):
        """
        user_emb: (embedding_dim,)
        item_hist_embs: (num_items_hist, embedding_dim)
        """
        Q = user_emb @ self.Wq  # (embedding_dim,)
        K = item_hist_embs @ self.Wk  # (num_items_hist, embedding_dim)
        V = item_hist_embs @ self.Wv  # (num_items_hist, embedding_dim)

        attn_scores = (Q @ K.T) / np.sqrt(self.embedding_dim)  # (num_items_hist,)
        attn_weights = np.exp(attn_scores - attn_scores.max())  # stability trick
        attn_weights /= attn_weights.sum()
        context = attn_weights @ V  # weighted sum → (embedding_dim,)
        return context

    def predict(self, user_ids, item_ids, user_histories):
        """
        user_histories: list of arrays, each array is item IDs the user has interacted with
        """
        preds = []
        for u, i, hist in zip(user_ids, item_ids, user_histories):
            user_emb = self.user_embeddings[u]
            item_emb = self.item_embeddings[i]
            hist_embs = self.item_embeddings[hist] if len(hist) > 0 else np.zeros((1, self.embedding_dim))

            context = self.attention(user_emb, hist_embs)
            x = np.concatenate([user_emb, context])
            y_pred = x @ self.W_out + self.b_out
            preds.append(y_pred)
        return np.array(preds).flatten()
    
    def save_checkpoint(self, path):
        """Save model parameters to a .npy file."""
        np.save(path, self.__dict__)

    def load_checkpoint(self, path):
        """Load model parameters from a saved .npy file."""
        checkpoint = np.load(path, allow_pickle=True).item()
        self.__dict__.update(checkpoint)
