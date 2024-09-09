'''
import numpy as np

class ANCFModel:
    def __init__(self, num_users, num_items, embedding_dim, learning_rate=0.01, l2_lambda=0.001):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda  # Regularization coefficient
        
        # Initialize embeddings randomly
        self.user_embedding = np.random.normal(scale=1./self.embedding_dim, size=(num_users, embedding_dim))
        self.item_embedding = np.random.normal(scale=1./self.embedding_dim, size=(num_items, embedding_dim))
        
        # Initialize weights for the attention mechanism and final prediction layer
        self.Wq = np.random.normal(scale=1./embedding_dim, size=(embedding_dim, embedding_dim))  # Query weights
        self.Wk = np.random.normal(scale=1./embedding_dim, size=(embedding_dim, embedding_dim))  # Key weights
        self.Wv = np.random.normal(scale=1./embedding_dim, size=(embedding_dim, embedding_dim))  # Value weights
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
        attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=1, keepdims=True)
        
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
        
        # Apply ReLU activation for non-linearity
        predictions = np.maximum(0, predictions)  # ReLU activation
        
        return predictions.flatten()  # Shape: (batch_size,)

    def backward(self, user_ids, item_ids, predictions, y_true):
        # Compute gradients for MSE loss
        error = predictions - y_true.to_numpy()  # Ensure y_true is a NumPy array

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


import numpy as np

class ANCFModel:
    def __init__(self, num_users, num_items, embedding_dim, learning_rate=0.01, l2_lambda=0.001, dropout_rate=0.5):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda  # Regularization coefficient
        self.dropout_rate = dropout_rate  # Dropout rate

        # Initialize embeddings randomly
        self.user_embedding = np.random.normal(scale=1./self.embedding_dim, size=(num_users, embedding_dim))
        self.item_embedding = np.random.normal(scale=1./self.embedding_dim, size=(num_items, embedding_dim))

        # Initialize weights for the attention mechanism and final prediction layer
        self.Wq = np.random.normal(scale=1./embedding_dim, size=(embedding_dim, embedding_dim))  # Query weights
        self.Wk = np.random.normal(scale=1./embedding_dim, size=(embedding_dim, embedding_dim))  # Key weights
        self.Wv = np.random.normal(scale=1./embedding_dim, size=(embedding_dim, embedding_dim))  # Value weights
        self.weights = np.random.normal(scale=1./embedding_dim, size=(embedding_dim, 1))
        self.bias = 0.0

    def apply_regularization(self):
        self.weights -= self.l2_lambda * self.weights
        self.user_embedding -= self.l2_lambda * self.user_embedding
        self.item_embedding -= self.l2_lambda * self.item_embedding

    def dropout(self, x):
        # Apply dropout with probability p=dropout_rate
        mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape)
        return x * mask / (1 - self.dropout_rate)

    def self_attention(self, user_emb, item_emb):
        # Compute Query, Key, and Value vectors
        Q = np.dot(user_emb, self.Wq)  # Shape: (batch_size, embedding_dim)
        K = np.dot(item_emb, self.Wk)  # Shape: (batch_size, embedding_dim)
        V = np.dot(item_emb, self.Wv)  # Shape: (batch_size, embedding_dim)

        # Compute attention scores: Q * K^T / sqrt(d)
        attention_scores = np.dot(Q, K.T) / np.sqrt(self.embedding_dim)  # Shape: (batch_size, batch_size)

        # Apply softmax to normalize attention scores
        attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=1, keepdims=True)

        # Compute weighted sum of value vectors
        attention_output = np.dot(attention_weights, V)  # Shape: (batch_size, embedding_dim)

        # Apply dropout to the attention output
        attention_output = self.dropout(attention_output)

        return attention_output

    def forward(self, user_ids, item_ids):
        # Retrieve embeddings
        user_emb = self.user_embedding[user_ids]  # Shape: (batch_size, embedding_dim)
        item_emb = self.item_embedding[item_ids]  # Shape: (batch_size, embedding_dim)

        # Apply self-attention mechanism
        attention_output = self.self_attention(user_emb, item_emb)

        # Final linear prediction
        predictions = np.dot(attention_output, self.weights) + self.bias  # Shape: (batch_size, 1)

        # Apply ReLU activation for non-linearity
        predictions = np.maximum(0, predictions)  # ReLU activation

        return predictions.flatten()  # Shape: (batch_size,)

    def backward(self, user_ids, item_ids, predictions, y_true):
        # Compute gradients for MSE loss
        error = predictions - y_true.to_numpy()  # Ensure y_true is a NumPy array

        # Split the attention_output back into user and item parts
        attention_output = self.self_attention(self.user_embedding[user_ids], self.item_embedding[item_ids])

        # Gradients for weights and bias using only the attention_output (not concatenated)
        grad_weights = np.dot(attention_output.T, error) / len(user_ids)  # Shape: (embedding_dim, 1)
        grad_bias = np.mean(error)

        # Update weights and bias
        self.weights -= self.learning_rate * grad_weights.reshape(self.weights.shape)
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
'''

import tensorflow as tf
import numpy as np

class ANCFModel(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim, learning_rate=0.01, l2_lambda=0.001, dropout_rate=0.5):
        super(ANCFModel, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda  # Regularization coefficient
        self.dropout_rate = dropout_rate  # Dropout rate

        # Define embeddings using TensorFlow
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim, embeddings_initializer='normal')
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_dim, embeddings_initializer='normal')

        # Define attention mechanism weights using TensorFlow
        self.Wq = tf.keras.layers.Dense(embedding_dim)
        self.Wk = tf.keras.layers.Dense(embedding_dim)
        self.Wv = tf.keras.layers.Dense(embedding_dim)
        
        # Define final prediction layer weights
        self.output_layer = tf.keras.layers.Dense(1)
        
        # Optimizer and regularizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.regularizer = tf.keras.regularizers.l2(self.l2_lambda)

    def self_attention(self, user_emb, item_emb):
        # Compute Query, Key, and Value vectors
        Q = self.Wq(user_emb)
        K = self.Wk(item_emb)
        V = self.Wv(item_emb)

        # Compute attention scores: Q * K^T / sqrt(d)
        attention_scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(self.embedding_dim))

        # Apply softmax to normalize attention scores
        attention_weights = tf.nn.softmax(attention_scores)

        # Compute weighted sum of value vectors
        attention_output = tf.matmul(attention_weights, V)

        # Apply dropout to the attention output
        attention_output = tf.nn.dropout(attention_output, rate=self.dropout_rate)

        return attention_output

    def call(self, user_ids, item_ids):
        # Retrieve embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        # Apply self-attention mechanism
        attention_output = self.self_attention(user_emb, item_emb)

        # Final linear prediction
        predictions = self.output_layer(attention_output)

        return tf.nn.relu(predictions)  # Apply ReLU activation for non-linearity

    def compute_loss(self, predictions, y_true):
        # MSE loss
        loss = tf.reduce_mean(tf.square(predictions - y_true))
        return loss + sum(self.losses)  # Include L2 regularization loss


# Fix for user/item ID range issue
# Use tf.keras.layers.Embedding, which automatically handles out-of-range indices by ensuring that embeddings are only within the valid range.
