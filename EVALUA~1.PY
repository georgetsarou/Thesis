import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import argparse
import joblib
from tensorflow.keras.models import load_model

def load_test_data():
    print("Loading test data...")
    X_test = np.load('data/processed/test/X_test.npy')
    y_test = pd.read_csv('data/processed/test/y_test.csv').values.ravel()
    label_mapping = pd.read_csv('data/processed/label_mapping.csv')
    label_to_index = {label: idx for label, idx in zip(label_mapping['label'], label_mapping['index'])}
    y_test_encoded = np.array([label_to_index[label] for label in y_test])
    
    print(f"Loaded test data with shape: {X_test.shape}")
    
    return X_test, y_test, y_test_encoded, label_mapping

def load_model_and_params():
    print("Loading trained model...")
    model_path = 'data/model/final_model.h5'
    model = load_model(model_path)
    params_df = pd.read_csv('data/model/model_params.csv')
    params = params_df.to_dict('records')[0]
    
    print(f"Loaded model from {model_path}")
    
    return model, params

def evaluate_model(model, X_test, y_test_encoded):
    print("Evaluating model on test data...")
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    test_loss, test_acc = model.evaluate(X_test, y_test_encoded, verbose=0)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_encoded, y_pred, average='weighted'
    )
    
    print("\n--- Model Evaluation Results ---")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")
    
    return y_pred, y_pred_proba, test_acc, test_loss, precision, recall, f1

def generate_classification_report(y_test_encoded, y_pred, label_mapping):
    print("\n--- Detailed Classification Report ---")
    unique_classes = sorted(set(y_test_encoded).union(set(y_pred)))
    idx_to_label = {idx: label for idx, label in zip(label_mapping['index'], label_mapping['label'])}
    class_names = [idx_to_label[i] for i in unique_classes if i in idx_to_label]
    report = classification_report(
        y_test_encoded, y_pred, 
        output_dict=True
    )
    print(classification_report(y_test_encoded, y_pred))
    report_df = pd.DataFrame(report).transpose()
    report_path = 'data/model/classification_report.csv'
    report_df.to_csv(report_path)
    print(f"Classification report saved to {report_path}")
    
    return report, report_df

def plot_confusion_matrix(y_test_encoded, y_pred, label_mapping):
    print("Generating confusion matrix visualization...")
    idx_to_label = {idx: label for idx, label in zip(label_mapping['index'], label_mapping['label'])}
    num_classes = len(idx_to_label)
    
    if num_classes > 20:
        cm = confusion_matrix(y_test_encoded, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, cmap='Blues')
        plt.title('Confusion Matrix (Labels omitted due to high number of classes)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
    else:
        class_names = [idx_to_label[i] for i in range(num_classes)]
        cm = confusion_matrix(y_test_encoded, y_pred)
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
    os.makedirs('data/model/figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig('data/model/figures/confusion_matrix.png', dpi=300)
    
    print("Confusion matrix saved to data/model/figures/confusion_matrix.png")
    
    return cm

def plot_class_distribution(y_test, y_pred, label_mapping):
    print("Generating class distribution visualization...")
    idx_to_label = {idx: label for idx, label in zip(label_mapping['index'], label_mapping['label'])}
    y_pred_labels = [idx_to_label[pred] for pred in y_pred]
    df_comp = pd.DataFrame({
        'True': y_test,
        'Predicted': y_pred_labels
    })
    plt.figure(figsize=(15, 8))
    true_counts = df_comp['True'].value_counts().sort_values(ascending=False)
    predicted_counts = df_comp['Predicted'].value_counts()
    if len(true_counts) > 20:
        top_classes = true_counts.head(20).index
        true_subset = true_counts[true_counts.index.isin(top_classes)]
        pred_subset = predicted_counts[predicted_counts.index.isin(top_classes)]
        x = np.arange(len(top_classes))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.bar(x - width/2, [true_subset.get(cls, 0) for cls in top_classes], width, label='True')
        ax.bar(x + width/2, [pred_subset.get(cls, 0) for cls in top_classes], width, label='Predicted')
        
        ax.set_xticks(x)
        ax.set_xticklabels(top_classes, rotation=90)
        ax.legend()
        
        plt.title('Distribution of Top 20 Classes: True vs Predicted')
    else:

        all_classes = sorted(set(list(true_counts.index) + list(predicted_counts.index)))
        x = np.arange(len(all_classes))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.bar(x - width/2, [true_counts.get(cls, 0) for cls in all_classes], width, label='True')
        ax.bar(x + width/2, [predicted_counts.get(cls, 0) for cls in all_classes], width, label='Predicted')
        
        ax.set_xticks(x)
        ax.set_xticklabels(all_classes, rotation=90)
        ax.legend()
        
        plt.title('Distribution of All Classes: True vs Predicted')
    
    plt.tight_layout()
    plt.savefig('data/model/figures/class_distribution.png', dpi=300)
    print("Class distribution saved to data/model/figures/class_distribution.png")

def plot_training_history():
    print("Plotting training history...")
    history_df = pd.read_csv('data/model/training_history.csv')
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_df['accuracy'], label='Training Accuracy')
    plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history_df['loss'], label='Training Loss')
    plt.plot(history_df['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('data/model/figures/training_history.png', dpi=300)
    print("Training history plot saved to data/model/figures/training_history.png")

def save_evaluation_summary(test_acc, test_loss, precision, recall, f1):
    print("Saving evaluation summary...")
    
    summary = {
        'Test Accuracy': test_acc,
        'Test Loss': test_loss,
        'Weighted Precision': precision,
        'Weighted Recall': recall,
        'Weighted F1-Score': f1
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('data/model/evaluation_summary.csv', index=False)
    print("Evaluation summary saved to data/model/evaluation_summary.csv")

def main():
    parser = argparse.ArgumentParser(description='GTD Model Evaluation')
    parser.add_argument('--model_path', type=str, default='data/model/final_model.h5',
                        help='Path to the trained model')
    args = parser.parse_args()
    X_test, y_test, y_test_encoded, label_mapping = load_test_data()
    model, params = load_model_and_params()
    y_pred, y_pred_proba, test_acc, test_loss, precision, recall, f1 = evaluate_model(
        model, X_test, y_test_encoded
    )
    report, report_df = generate_classification_report(y_test_encoded, y_pred, label_mapping)
    cm = plot_confusion_matrix(y_test_encoded, y_pred, label_mapping)
    plot_class_distribution(y_test, y_pred, label_mapping)
    plot_training_history()
    save_evaluation_summary(test_acc, test_loss, precision, recall, f1)
    
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()