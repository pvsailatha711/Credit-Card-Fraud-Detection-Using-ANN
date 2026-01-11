# Credit Card Fraud Detection Using Artificial Neural Networks

## Overview

This project implements a binary classification system to detect fraudulent credit card transactions using Artificial Neural Networks (ANN). The model is trained on a highly imbalanced dataset containing over 284,000 transactions, where fraudulent cases represent only 0.17% of the total data.

## Dataset

The project uses the Credit Card Fraud Detection dataset, which contains the following characteristics:

- **Total Transactions**: 284,807
- **Features**: 31 columns
  - `Time`: Seconds elapsed between each transaction and the first transaction
  - `V1-V28`: Principal components obtained through PCA transformation (for confidentiality)
  - `Amount`: Transaction amount
  - `Class`: Target variable (0 = legitimate, 1 = fraudulent)
- **Class Distribution**:
  - Legitimate transactions: 284,315 (99.83%)
  - Fraudulent transactions: 492 (0.17%)

The dataset contains no missing values, making it ready for preprocessing and model training.

## Methodology

### Data Preprocessing

1. **Exploratory Data Analysis**: Initial analysis to understand feature distributions and class imbalance
2. **Feature Scaling**: StandardScaler applied to normalize features and ensure optimal neural network convergence
3. **Train-Test Split**: Dataset divided into training and testing sets for model validation

### Model Architecture

The neural network is built using Keras Sequential API with the following structure:

- **Input Layer**: 30 features
- **Hidden Layer 1**: 15 neurons with ReLU activation
- **Hidden Layer 2**: 15 neurons with ReLU activation
- **Output Layer**: 1 neuron with Sigmoid activation for binary classification
- **Weight Initialization**: Uniform kernel initializer

### Training Configuration

- **Epochs**: 50
- **Optimizer**: Adam (implied from standard binary classification setup)
- **Loss Function**: Binary crossentropy
- **Batch Processing**: Configured for efficient training on large dataset

## Results

The model demonstrates strong fraud detection capabilities with the following performance metrics:

### Overall Performance
- **Accuracy**: 84.38%

### Confusion Matrix
|                    | Predicted Normal | Predicted Fraud |
|--------------------|------------------|-----------------|
| **Actual Normal**  | 59,967          | 11,115          |
| **Actual Fraud**   | 7               | 113             |

### Classification Metrics (Fraud Class)
- **Recall**: 94% - Successfully identifies 94% of actual fraud cases
- **Precision**: 1% - High false positive rate
- **F1-Score**: 0.02

## Key Findings

The model excels at identifying fraudulent transactions with a 94% recall rate, meaning it catches nearly all fraud cases. However, the low precision (1%) indicates a significant number of false positives, where legitimate transactions are flagged as fraudulent. This trade-off is common in fraud detection systems where missing actual fraud is more costly than investigating false alarms.

### Practical Implications

- **Strengths**: Only 7 fraudulent transactions out of 120 were missed, demonstrating excellent fraud detection capability
- **Limitations**: 11,115 legitimate transactions were incorrectly flagged, which could impact customer experience
- **Use Case**: This model is suitable for initial fraud screening where high sensitivity is prioritized, with flagged transactions undergoing secondary verification

## Technologies Used

- **Python**: Core programming language
- **TensorFlow/Keras**: Deep learning framework for building and training the ANN
- **Scikit-learn**: Data preprocessing and evaluation metrics
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Data visualization

## Acknowledgments

Dataset sourced from Kaggle's Credit Card Fraud Detection dataset, containing anonymized credit card transactions from European cardholders.
