import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold, StratifiedKFold
from datetime import datetime
import argparse
import joblib

def load_training_data():
    print("Loading training data...")
    
    X_train = np.load('data/processed/train/X_train.npy')
    y_train = pd.read_csv('data/processed/train/y_train.csv').values.ravel()
    label_mapping = pd.read_csv('data/processed/label_mapping.csv')
    label_to_index = {label: idx for idx, label in zip(label_mapping['index'], label_mapping['label'])}
    y_train_encoded = np.array([label_to_index[label] for label in y_train])
    
    num_classes = len(label_mapping)
    input_dim = X_train.shape[1]
    
    print(f"Loaded training data with shape: {X_train.shape}")
    print(f"Number of target classes: {num_classes}")
    
    return X_train, y_train_encoded, num_classes, input_dim, label_mapping

def create_model(input_dim, num_classes, params):
    print("Creating model architecture...")
    
    model = Sequential()
    model.add(Dense(params['units_layer1'], 
                   activation=params['activation'], 
                   input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(params['units_layer2'], activation=params['activation']))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout_rate']))
    
    model.add(Dense(params['units_layer3'], activation=params['activation']))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(
        optimizer=Adam(learning_rate=params['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def setup_callbacks(fold=None):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/{current_time}"
    if fold is not None:
        log_dir = f"{log_dir}/fold_{fold}"
        model_path = f"data/model/model_fold_{fold}.h5"
    else:
        model_path = f"data/model/model_final.h5"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("data/model", exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch=0
        )
    ]
    
    return callbacks

from sklearn.model_selection import StratifiedKFold

def train_with_kfold(X_train, y_train_encoded, num_classes, input_dim, params):
    print(f"Starting {params['n_folds']}-fold cross validation training...")
    
    kfold = StratifiedKFold(n_splits=params['n_folds'], shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train_encoded)):
        print(f"\n--- Training Fold {fold+1}/{params['n_folds']} ---")
        
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train_encoded[train_idx], y_train_encoded[val_idx]
        
        train_classes = set(y_fold_train)
        val_classes = set(y_fold_val)
        print(f"Classes in fold training set: {len(train_classes)}")
        print(f"Classes in fold validation set: {len(val_classes)}")
        
        model = create_model(input_dim, num_classes, params)
        
        callbacks = setup_callbacks(fold+1)
        
        history = model.fit(
            X_fold_train, y_fold_train,
            validation_data=(X_fold_val, y_fold_val),
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            verbose=1,
            callbacks=callbacks
        )
        
        val_loss, val_acc = model.evaluate(X_fold_val, y_fold_val, verbose=0)
        fold_scores.append(val_acc)
        
        print(f"Fold {fold+1} - Validation Accuracy: {val_acc:.4f}")
    
    print("\n--- K-Fold Cross Validation Results ---")
    print(f"Mean Validation Accuracy: {np.mean(fold_scores):.4f}")
    print(f"Standard Deviation: {np.std(fold_scores):.4f}")
    
    return fold_scores

def train_final_model(X_train, y_train_encoded, num_classes, input_dim, params):
    print("\n--- Training Final Model on Full Training Data ---")
    

    model = create_model(input_dim, num_classes, params)
    

    callbacks = setup_callbacks()
    

    val_split = 0.1
    

    history = model.fit(
        X_train, y_train_encoded,
        validation_split=val_split,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        verbose=1,
        callbacks=callbacks
    )
    

    model.save('data/model/final_model.h5')
    

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('data/model/training_history.csv', index=False)
    
    print("Final model saved to data/model/final_model.h5")
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description='GTD Model Training')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--skip_kfold', action='store_true', help='Skip k-fold cross-validation')
    
    args = parser.parse_args()
    
    # Set up model parameters
    params = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'n_folds': args.n_folds,
        'units_layer1': 256,
        'units_layer2': 128,
        'units_layer3': 64,
        'activation': 'relu',
        'dropout_rate': 0.3
    }
    

    X_train, y_train_encoded, num_classes, input_dim, label_mapping = load_training_data()
    

    if not args.skip_kfold:
        fold_scores = train_with_kfold(X_train, y_train_encoded, num_classes, input_dim, params)
    

    model, history = train_final_model(X_train, y_train_encoded, num_classes, input_dim, params)
    

    params_df = pd.DataFrame([params])
    params_df.to_csv('data/model/model_params.csv', index=False)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()