import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import shap
import argparse
import joblib
from tensorflow.keras.models import load_model

def load_model_and_preprocessing():
    print("Loading model and preprocessing components...")
    model = load_model('data/model/final_model.h5')
    preprocessor = joblib.load('data/processed/preprocessor.joblib')
    feature_names = pd.read_csv('data/processed/feature_names.csv').values.ravel()
    label_mapping = pd.read_csv('data/processed/label_mapping.csv')
    
    print(f"Loaded model with {len(feature_names)} input features and {len(label_mapping)} output classes")
    
    return model, preprocessor, feature_names, label_mapping

def preprocess_data(data, preprocessor):
    print("Preprocessing inference data...")
    preprocessed_data = preprocessor.transform(data)
    print(f"Preprocessed data shape: {preprocessed_data.shape}")
    return preprocessed_data

def make_prediction(model, preprocessed_data, label_mapping):
    print("Making predictions...")
    predictions_proba = model.predict(preprocessed_data)
    predictions = np.argmax(predictions_proba, axis=1)
    idx_to_label = {idx: label for idx, label in zip(label_mapping['index'], label_mapping['label'])}
    predictions_labels = [idx_to_label[pred] for pred in predictions]
    confidence = np.max(predictions_proba, axis=1)
    results_df = pd.DataFrame({
        'Predicted_Class': predictions_labels,
        'Confidence': confidence
    })
    
    print("Prediction results summary:")
    print(results_df.head())
    
    return predictions, predictions_proba, results_df

def calculate_shap_values(model, preprocessed_data, feature_names, sample_size=100):
    print("Calculating SHAP values for model explanation...")
    if preprocessed_data.shape[0] > sample_size:
        print(f"Using a subset of {sample_size} samples for SHAP analysis")
        indices = np.random.choice(preprocessed_data.shape[0], sample_size, replace=False)
        data_sample = preprocessed_data[indices]
    else:
        data_sample = preprocessed_data
    explainer = shap.DeepExplainer(model, data_sample)
    shap_values = explainer.shap_values(data_sample)
    
    print(f"Generated SHAP values with shape: {np.array(shap_values).shape}")
    
    return explainer, shap_values, data_sample

def plot_shap_summary(shap_values, features, feature_names, label_mapping, max_display=20):
    print("Generating SHAP summary plots...")
    os.makedirs('data/model/figures/shap', exist_ok=True)
    idx_to_label = {idx: label for idx, label in zip(label_mapping['index'], label_mapping['label'])}
    print(f"SHAP values shape: {np.array(shap_values).shape}")
    background = np.zeros((1, features.shape[1]))
    print("Generating global summary plot...")
    plt.figure(figsize=(12, 10))
    feature_names_list = feature_names.tolist() if isinstance(feature_names, np.ndarray) else list(feature_names)
    if len(feature_names_list) != features.shape[1]:
        feature_names_list = [f"Feature_{i}" for i in range(features.shape[1])]
    all_class_values = np.abs(np.array(shap_values)).mean(axis=0)
    feature_importance = np.mean(all_class_values, axis=0)
    top_indices = np.argsort(-feature_importance)[:max_display]
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_indices)), feature_importance[top_indices])
    plt.yticks(range(len(top_indices)), [feature_names_list[i] if i < len(feature_names_list) else f"Feature_{i}" for i in top_indices])
    plt.xlabel('Mean |SHAP value|')
    plt.title('Top Features (Global Importance)')
    plt.tight_layout()
    plt.savefig('data/model/figures/shap/global_feature_importance.png', dpi=300)
    plt.close()
    
    print("Saved global feature importance plot to data/model/figures/shap/global_feature_importance.png")
    print("Generating class-specific summary plots...")
    class_importance = [np.abs(shap_values[i]).sum() for i in range(len(shap_values))]
    top_classes = np.argsort(class_importance)[-5:]
    
    for class_idx in top_classes:
        if class_idx >= len(idx_to_label):
            continue
        class_name = idx_to_label.get(class_idx, f"Class_{class_idx}")
        print(f"Generating summary plot for class: {class_name}")
        class_shap_values = shap_values[class_idx]
        feature_importance = np.mean(np.abs(class_shap_values), axis=0)
        top_indices = np.argsort(-feature_importance)[:max_display]
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_indices)), feature_importance[top_indices])
        plt.yticks(range(len(top_indices)), [feature_names_list[i] if i < len(feature_names_list) else f"Feature_{i}" for i in top_indices])
        plt.xlabel('Mean |SHAP value|')
        plt.title(f'Top Features for Class: {class_name}')
        plt.tight_layout()
        safe_class_name = "".join(c if c.isalnum() else "_" for c in class_name)
        plt.savefig(f'data/model/figures/shap/class_{class_idx}_{safe_class_name}_importance.png', dpi=300)
        plt.close()
        
        print(f"Saved feature importance plot for class {class_name}")

def plot_shap_force_plots(explainer, shap_values, data_sample, feature_names, label_mapping, num_examples=5):
    print(f"Skipping SHAP force plots due to dimensionality issues")
    os.makedirs('data/model/feature_importance', exist_ok=True)
    global_shap_values = np.abs(np.array(shap_values)).mean(axis=0)
    mean_importance = global_shap_values.mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature_Index': range(len(mean_importance)),
        'Mean_Importance': mean_importance
    })
    importance_df = importance_df.sort_values('Mean_Importance', ascending=False)
    importance_df.to_csv('data/model/feature_importance/global_feature_importance.csv', index=False)
    print("Saved feature importance data to data/model/feature_importance/global_feature_importance.csv")

def save_feature_importance(shap_values, feature_names, label_mapping):
    print("Saving feature importance information...")
    os.makedirs('data/model/feature_importance', exist_ok=True)
    global_shap_values = np.abs(np.array(shap_values)).mean(axis=0)
    mean_importance = global_shap_values.mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature_Index': range(len(mean_importance)),
        'Mean_Importance': mean_importance
    })
    importance_df = importance_df.sort_values('Mean_Importance', ascending=False)
    importance_df.to_csv('data/model/feature_importance/global_feature_importance.csv', index=False)
    print("Saved feature importance data to data/model/feature_importance/global_feature_importance.csv")

def main():
    parser = argparse.ArgumentParser(description='GTD Model Inference and Explainability')
    parser.add_argument('--data_path', type=str, 
                        help='Path to the data file for inference (CSV or Excel)')
    parser.add_argument('--num_examples', type=int, default=5,
                        help='Number of examples to generate force plots for')
    args = parser.parse_args()
    model, preprocessor, feature_names, label_mapping = load_model_and_preprocessing()
    if args.data_path is None:
        print("No data path provided. Using test data.")
        X_data = np.load('data/processed/test/X_test.npy')
        preprocessed_data = X_data
    else:
        if args.data_path.endswith('.csv'):
            data = pd.read_csv(args.data_path)
        elif args.data_path.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(args.data_path)
        else:
            raise ValueError("Unsupported file format. Please provide CSV or Excel file.")
        preprocessed_data = preprocess_data(data, preprocessor)
    predictions, predictions_proba, results_df = make_prediction(model, preprocessed_data, label_mapping)
    os.makedirs('data/model/predictions', exist_ok=True)
    results_df.to_csv('data/model/predictions/prediction_results.csv', index=False)
    print("Prediction results saved to data/model/predictions/prediction_results.csv")
    explainer, shap_values, data_sample = calculate_shap_values(model, preprocessed_data, feature_names)
    plot_shap_summary(shap_values, data_sample, feature_names, label_mapping)
    plot_shap_force_plots(explainer, shap_values, data_sample, feature_names, label_mapping, 
                         num_examples=args.num_examples)
    save_feature_importance(shap_values, feature_names, label_mapping)
    
    print("\nInference and explainability analysis completed successfully!")

if __name__ == "__main__":
    main()