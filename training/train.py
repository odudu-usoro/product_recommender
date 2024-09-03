import numpy as np
from ancf_models import ANCFModel
from preprocess import preprocess_data
from dataloader import batch_data

def train_model(model, train_batches, val_batches, epochs, learning_rate):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_batches:
            user_ids = X_batch['user_id'].values
            item_ids = X_batch['product_id'].values
            
            # Forward pass
            predictions = model.forward(user_ids, item_ids)
            loss = np.mean((predictions - y_batch) ** 2)  # MSE loss
            train_loss += loss
            
            # Backpropagation and optimization
            model.backward()  # Define this function to handle gradient updates
            model.update_weights(learning_rate)
            
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {train_loss/len(train_batches)}')

        # Validate on validation set
        val_loss = validate_model(model, val_batches)
        print(f'Validation Loss: {val_loss}')
        
def validate_model(model, val_batches):
    model.eval()
    val_loss = 0
    with np.no_grad():
        for X_batch, y_batch in val_batches:
            user_ids = X_batch['user_id'].values
            item_ids = X_batch['product_id'].values
            predictions = model.forward(user_ids, item_ids)
            loss = np.mean((predictions - y_batch) ** 2)  # MSE loss
            val_loss += loss
    return val_loss / len(val_batches)
