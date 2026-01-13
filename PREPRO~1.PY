import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import argparse

def create_directory_structure():
    dirs = [
        'data/raw',
        'data/processed/train',
        'data/processed/test',
        'data/model'
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def load_data(data_path):
    print(f"Loading data from {data_path}...")
    
    try:
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path, low_memory=False)
        elif data_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(data_path)
        else:
            raise ValueError("Unsupported file format. Please provide CSV or Excel file.")
        
        print(f"Loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def filter_european_countries(df):
    print("Filtering for European countries only...")
    

    european_regions = [
        8,
        9 
    ]
    
    europe_df = df[df['region'].isin(european_regions)].copy()
    
    print(f"Filtered data shape: {europe_df.shape}")
    print(f"Data covers {europe_df['country_txt'].nunique()} European countries")
    
    return europe_df

def create_target_variable(df):
    print("Creating combined attack-target type variable...")
    
    df['attack_target_type'] = df['attacktype1_txt'] + '_' + df['targtype1_txt']
    
    target_counts = df['attack_target_type'].value_counts()
    print(f"Created {len(target_counts)} unique attack-target combinations")
    print(f"Top 5 most common combinations:\n{target_counts.head()}")
    
    valid_classes = target_counts[target_counts >= 6].index
    df = df[df['attack_target_type'].isin(valid_classes)]
    
    filtered_counts = df['attack_target_type'].value_counts()
    print(f"After filtering, {len(filtered_counts)} classes remain with at least 6 samples each")
    
    return df

def handle_missing_values(df):
    print("Handling missing values...")
    missing_before = df.isna().sum().sum()
    print(f"Missing values before imputation: {missing_before}")
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    missing_after = df.isna().sum().sum()
    print(f"Missing values after imputation: {missing_after}")
    
    return df

def drop_unnecessary_features(df):
    print("Dropping unnecessary features...")
    features_to_drop = [
        'attacktype1', 'attacktype1_txt', 'targtype1', 'targtype1_txt',
        'attacktype2', 'attacktype2_txt', 'attacktype3', 'attacktype3_txt',
        'targtype2', 'targtype2_txt', 'targtype3', 'targtype3_txt',
        'eventid', 'approxdate', 'resolution',
        'summary', 'motive', 'addnotes', 'scite1', 'scite2', 'scite3',
        'propcomment', 'location', 'ransomnote',
        'country_txt', 'region_txt',
        'corp1', 'corp2', 'corp3', 'target1', 'target2', 'target3'
    ]
    
    features_to_drop = [f for f in features_to_drop if f in df.columns]
    original_feature_count = df.shape[1]
    df = df.drop(columns=features_to_drop, errors='ignore')
    new_feature_count = df.shape[1]
    
    print(f"Dropped {original_feature_count - new_feature_count} features")
    print(f"Remaining features: {new_feature_count}")
    
    return df

def encode_features(df, target_col='attack_target_type'):
    print("Encoding features...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Numerical features: {len(numerical_cols)}")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_cols),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_cols)
        ]
    )
    X_transformed = preprocessor.fit_transform(X)
    onehot_cols = []
    for i, col in enumerate(categorical_cols):
        categories = preprocessor.transformers_[1][1].named_steps['onehot'].categories_[i]
        for category in categories:
            onehot_cols.append(f"{col}_{category}")
    
    transformed_cols = numerical_cols + onehot_cols
    
    print(f"Data shape after transformation: {X_transformed.shape}")
    
    return X_transformed, y, preprocessor, transformed_cols

def handle_class_imbalance(X, y):
    print("Handling class imbalance...")
    
    class_counts_before = pd.Series(y).value_counts()
    print(f"Class distribution before balancing (top 5):\n{class_counts_before.head()}")
    
    if X.shape[1] > 5000:
        print(f"Skipping SMOTE due to high dimensionality ({X.shape[1]} features)")
        return X, y
    
    try:
        smote = SMOTE(random_state=42, k_neighbors=1)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        class_counts_after = pd.Series(y_resampled).value_counts()
        print(f"Class distribution after balancing: {len(class_counts_after)} classes with {class_counts_after.iloc[0]} samples each")
        print(f"Data shape after balancing: {X_resampled.shape}")
        
        return X_resampled, y_resampled
    except ValueError as e:
        print(f"Warning: Could not apply SMOTE due to: {e}")
        print("Continuing with imbalanced dataset")
        return X, y
    except MemoryError:
        print("Warning: Not enough memory to apply SMOTE")
        print("Continuing with imbalanced dataset")
        return X, y

def split_train_test(X, y, test_size=0.2):
    print(f"Splitting data into train ({1-test_size:.0%}) and test ({test_size:.0%}) sets...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    train_classes = set(y_train)
    test_classes = set(y_test)
    all_classes = set(y)
    
    print(f"Total unique classes: {len(all_classes)}")
    print(f"Classes in training set: {len(train_classes)}")
    print(f"Classes in test set: {len(test_classes)}")
    
    missing_in_train = all_classes - train_classes
    missing_in_test = all_classes - test_classes
    
    if missing_in_train:
        print(f"Warning: {len(missing_in_train)} classes missing from training set")
    if missing_in_test:
        print(f"Warning: {len(missing_in_test)} classes missing from test set")
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test, preprocessor, transformed_cols):
    print("Saving processed data...")
    train_dir = 'data/processed/train'
    np.save(os.path.join(train_dir, 'X_train.npy'), X_train)
    pd.Series(y_train).to_csv(os.path.join(train_dir, 'y_train.csv'), index=False)
    test_dir = 'data/processed/test'
    np.save(os.path.join(test_dir, 'X_test.npy'), X_test)
    pd.Series(y_test).to_csv(os.path.join(test_dir, 'y_test.csv'), index=False)
    import joblib
    joblib.dump(preprocessor, 'data/processed/preprocessor.joblib')
    pd.Series(transformed_cols).to_csv('data/processed/feature_names.csv', index=False)
    label_mapping = {i: label for i, label in enumerate(sorted(set(y_train)))}
    pd.DataFrame({
        'index': label_mapping.keys(),
        'label': label_mapping.values()
    }).to_csv('data/processed/label_mapping.csv', index=False)
    
    print("Data successfully saved to processed directories")

def main():
    parser = argparse.ArgumentParser(description='GTD Data Preprocessing')
    parser.add_argument('--data_path', type=str, default='data/raw/globalterrorismdb_0522dist.xlsx',
                       help='Path to the raw GTD data file')
    args = parser.parse_args()
    create_directory_structure()
    df = load_data(args.data_path)
    europe_df = filter_european_countries(df)
    europe_df = create_target_variable(europe_df)
    europe_df = handle_missing_values(europe_df)
    europe_df = drop_unnecessary_features(europe_df)
    X_transformed, y, preprocessor, transformed_cols = encode_features(europe_df)
    X_resampled, y_resampled = handle_class_imbalance(X_transformed, y)
    X_train, X_test, y_train, y_test = split_train_test(X_resampled, y_resampled)
    save_processed_data(X_train, X_test, y_train, y_test, preprocessor, transformed_cols)

if __name__ == "__main__":
    main()