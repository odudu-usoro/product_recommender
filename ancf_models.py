import numpy as np

class ANCFModel:
    def __init__(self, num_users, num_items, embedding_dim, learning_rate=0.01):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        
        # Initialize embeddings randomly
        self.user_embedding = np.random.normal(scale=1./self.embedding_dim, size=(num_users, embedding_dim))
        self.item_embedding = np.random.normal(scale=1./self.embedding_dim, size=(num_items, embedding_dim))
        
        # Initialize weights for the network (simple linear layer for demonstration)
        self.weights = np.random.normal(scale=1./embedding_dim, size=(2 * embedding_dim, 1))
        self.bias = 0.0

    def train_mode(self):
        pass  # Placeholder for compatibility

    def eval_mode(self):
        pass  # Placeholder for compatibility

    def forward(self, user_ids, item_ids):
        # Retrieve embeddings
        user_emb = self.user_embedding[user_ids]  # Shape: (batch_size, embedding_dim)
        item_emb = self.item_embedding[item_ids]  # Shape: (batch_size, embedding_dim)
        
        # Concatenate user and item embeddings
        combined = np.concatenate([user_emb, item_emb], axis=1)  # Shape: (batch_size, 2 * embedding_dim)
        
        # Simple linear prediction
        predictions = np.dot(combined, self.weights) + self.bias  # Shape: (batch_size, 1)
        return predictions.flatten()  # Shape: (batch_size,)

    def backward(self, user_ids, item_ids, predictions, y_true):
        # Compute gradients for a simple linear model with MSE loss
        error = predictions - y_true  # Shape: (batch_size,)
        grad_weights = np.dot(np.concatenate([self.user_embedding[user_ids], self.item_embedding[item_ids]], axis=1).T, error[:, np.newaxis]) / len(user_ids)
        grad_bias = np.mean(error)
        
        # Update weights and bias
        self.weights -= self.learning_rate * grad_weights
        self.bias -= self.learning_rate * grad_bias
        
        # Compute gradients for embeddings
        grad_combined = np.outer(error, self.weights.flatten())  # Shape: (2 * embedding_dim,)
        grad_user = grad_combined[:self.embedding_dim]
        grad_item = grad_combined[self.embedding_dim:]
        
        # Update embeddings
        self.user_embedding[user_ids] -= self.learning_rate * grad_user
        self.item_embedding[item_ids] -= self.learning_rate * grad_item

    def update_weights(self, learning_rate):
        pass  # Already handled in backward

    def backward(self, user_ids, item_ids, predictions, y_true):
        # Compute gradients for a simple linear model with MSE loss
        error = predictions - y_true  # Shape: (batch_size,)
        combined = np.concatenate([self.user_embedding[user_ids], self.item_embedding[item_ids]], axis=1)  # Shape: (batch_size, 2 * embedding_dim)
        
        grad_weights = np.dot(combined.T, error) / len(user_ids)  # Shape: (2 * embedding_dim, )
        grad_bias = np.mean(error)
        
        # Update weights and bias
        self.weights -= self.learning_rate * grad_weights.reshape(self.weights.shape)
        self.bias -= self.learning_rate * grad_bias
        
        # Gradients for embeddings
        grad_combined = np.dot(error.to_numpy().reshape(-1, 1), self.weights.T)  # Shape: (batch_size, 2 * embedding_dim)
        grad_user = grad_combined[:, :self.embedding_dim]
        grad_item = grad_combined[:, self.embedding_dim:]
        
        # Update embeddings
        self.user_embedding[user_ids] -= self.learning_rate * grad_user
        self.item_embedding[item_ids] -= self.learning_rate * grad_item
