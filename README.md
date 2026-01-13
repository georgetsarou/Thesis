# GTD-Attack-Prediction: Advanced Classification of Terrorist Incidents in Europe

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.8+-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Project Overview

This repository implements a comprehensive machine learning solution to predict attack and target types in terrorist incidents across European countries using the Global Terrorism Database (GTD). The project demonstrates advanced data preprocessing, deep learning model development, evaluation metrics, and explainable AI techniques in a counterterrorism context. The model aims to predict a combined target variable that represents both the attack type (e.g., bombing, assassination) and the target type (e.g., government, civilians, military) for terrorist incidents, providing valuable insights for security analysts and policymakers.

## Dataset

The [Global Terrorism Database (GTD)](https://www.start.umd.edu/data-tools/GTD) is an open-source database including information on terrorist attacks around the world from 1970 through the present. It contains over 200,000 incidents with more than 100 variables per event, including:

- Temporal information (date, time)
- Spatial information (location, country)
- Attack information (type, weapons, casualties)
- Target information (type, nationality)
- Group information (name, claimed responsibility)

For this project, we focus exclusively on terrorist incidents in European countries.

## Repository Structure

```
GTD-Attack-Prediction/
│
├── data/                           # Data directory
│   ├── raw/                        # Raw GTD data (user-provided)
│   ├── processed/                  # Processed datasets
│   │   ├── train/                  # Training data
│   │   └── test/                   # Test data
│   └── model/                      # Saved model files
│
├── src/                            # Source code
│   ├── preprocessing.py            # Data preprocessing pipeline
│   ├── train.py                    # Model training with k-fold validation
│   ├── evaluation.py               # Model evaluation on test set
│   └── inference_xai.py            # Inference and explainability analysis
│
├── .gitignore                      # Git ignore file
├── LICENSE                         # MIT License
├── README.md                       # Project documentation
└── requirements.txt                # Project dependencies
```

## Methodology

### 1. Data Preprocessing (`preprocessing.py`)

The preprocessing pipeline performs several key steps to prepare the GTD data for modeling:

#### Data Filtering and Target Variable Creation
- Filters incidents to include only European countries (Western and Eastern Europe)
- Creates a novel combined target variable by concatenating the attack type and target type
- Filters classes to ensure each has at least 6 samples for statistical validity
- Results in a balanced set of attack-target type combinations

#### Data Cleaning and Feature Engineering
- Handles missing values using appropriate strategies for different data types:
  - Categorical variables: filled with 'Unknown'
  - Numerical variables: filled with median values
- Removes redundant or irrelevant features to reduce dimensionality
- Preserves temporal, geographical, and incident-specific information

#### Feature Encoding and Transformation
- Implements a preprocessing pipeline using scikit-learn's ColumnTransformer
- Numerical features: standardized using StandardScaler
- Categorical features: one-hot encoded with handling for unknown categories
- Preserves feature semantics while making data suitable for deep learning

#### Class Imbalance Handling
- Assesses class distribution in the target variable
- For high-dimensional data, uses strategic undersampling of majority classes
- For lower-dimensional data, applies Synthetic Minority Over-sampling Technique (SMOTE)
- Ensures balanced class representation for robust model training

#### Train-Test Splitting
- Implements stratified 80-20 train-test split to maintain class distribution
- Verifies class representation in both training and test sets
- Ensures no data leakage between training and test sets

### 2. Model Training (`train.py`)

The training module implements a sophisticated deep learning approach with cross-validation:

#### Model Architecture
- Deep neural network with multiple dense layers
- Layer 1: 256 neurons with ReLU activation
- Layer 2: 128 neurons with ReLU activation
- Layer 3: 64 neurons with ReLU activation
- Output layer: Softmax activation for multi-class classification

#### Regularization Techniques
- Batch normalization after each hidden layer to stabilize learning
- Dropout (30%) to prevent overfitting
- Early stopping based on validation loss with patience of 10 epochs

#### Training Strategy
- Implements k-fold cross-validation (k=5) using StratifiedKFold
- Ensures balanced class representation in each fold
- Adam optimizer with learning rate of 0.001
- Sparse categorical cross-entropy loss function
- Batch size of 64 and maximum 100 epochs

#### Monitoring and Checkpointing
- TensorBoard integration for real-time monitoring of training metrics
- ModelCheckpoint to save the best model based on validation loss
- Comprehensive training history saved for later analysis

#### Final Model Training
- After cross-validation, trains final model on full training dataset
- Uses 10% validation split for early stopping
- Saves model, parameters, and training history

### 3. Model Evaluation (`evaluation.py`)

The evaluation module provides performance assessment:

#### Performance Metrics
- Accuracy: Overall correctness of predictions
- Loss: Model's prediction error
- Precision: Prediction correctness when a class is predicted
- Recall: Ability to find all instances of a class
- F1-Score: Harmonic mean of precision and recall

#### Detailed Analysis
- Class-specific metrics for each attack-target combination
- Confusion matrix visualization to identify common misclassifications
- Class distribution analysis comparing true vs. predicted distributions

#### Visualization
- Training and validation metrics over epochs
- Confusion matrix heatmap
- Class distribution comparison
- Results saved as high-quality figures for reporting

#### Output
- Comprehensive classification report
- Performance summary
- Visualizations for model behavior analysis

### 4. Inference and Explainability (`inference_xai.py`)

The inference module enables prediction on new data and provides explainable AI insights:

#### Inference Capabilities
- Loads the trained model and preprocessing components
- Accepts new data in CSV or Excel format
- Applies the same preprocessing pipeline used during training
- Generates predictions with confidence scores

#### Explainable AI with SHAP
- Implements SHapley Additive exPlanations (SHAP)
- Calculates feature importance for model predictions
- Identifies top contributing features for each class
- Creates visualizations of feature importance

#### Feature Importance Analysis
- Global feature importance across all classes
- Class-specific feature importance for top classes
- Comprehensive feature rankings
- Saved as CSV files and visualizations

#### Visualization
- Feature importance bar charts
- Detailed rankings of feature contributions
- Analysis of model decision-making

## Some Key Findings

1. **Attack Type Patterns**: Bombing/Explosion consistently appears as the dominant attack method across multiple target types
2. **Target Distribution**: Business, Private Citizens, and Government targets emerge as the most frequent targets
3. **Feature Importance**: Geographical and temporal features are among the most influential for prediction
4. **Regional Variations**: Clear patterns in attack methods and targets appear across different European regions

## Usage Instructions

### Prerequisites
- Python 3.8+
- TensorFlow 2.8+
- scikit-learn 1.0+
- Additional requirements in `requirements.txt`

### Installation
```bash
# Clone the repository
git clone https://github.com/username/GTD-Attack-Prediction.git
cd GTD-Attack-Prediction

# Install dependencies
pip install -r requirements.txt

# Prepare data directory
mkdir -p data/raw
```

### Data Preparation
Place the Global Terrorism Database file (in CSV or Excel format) in the `data/raw/` directory.

### Running the Pipeline
```bash
# 1. Preprocess the data
python src/preprocessing.py --data_path data/raw/your_gtd_file.xlsx

# 2. Train the model
python src/train.py --epochs 100 --batch_size 64 --n_folds 5

# 3. Evaluate the model
python src/evaluation.py

# 4. Run inference and explainability analysis
python src/inference_xai.py
```

### Optional Parameters
- `preprocessing.py`:
  - `--data_path`: Path to the GTD data file (default: 'data/raw/globalterrorismdb_0522dist.xlsx')

- `train.py`:
  - `--epochs`: Maximum number of training epochs (default: 100)
  - `--batch_size`: Batch size for training (default: 64)
  - `--learning_rate`: Learning rate for optimizer (default: 0.001)
  - `--n_folds`: Number of folds for cross-validation (default: 5)
  - `--skip_kfold`: Skip k-fold cross-validation and train final model directly

- `evaluation.py`:
  - `--model_path`: Path to the trained model (default: 'data/model/final_model.h5')

- `inference_xai.py`:
  - `--data_path`: Path to data for inference (default: uses test data)
  - `--num_examples`: Number of examples for SHAP force plots (default: 5)


## Conclusions and Future Work

This project demonstrates the effectiveness of deep learning for predicting attack and target types in terrorist incidents. The high accuracy and detailed explainability analysis provide valuable insights for security professionals.

Potential future improvements include:
- Enhanced feature engineering to capture more complex patterns
- Experimentation with alternative model architectures (LSTM, Transformer)
- Incorporation of additional contextual data for improved predictions
- Development of a web interface for real-time inference and visualization

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The Global Terrorism Database (GTD) maintained by the National Consortium for the Study of Terrorism and Responses to Terrorism (START)
- The scikit-learn, TensorFlow, and SHAP open-source projects
- All contributors to the field of machine learning for security applications

---

*Note: This project is for academic and research purposes only. The aim is to advance understanding of patterns in terrorist incidents to support security professionals, policymakers, and researchers.*