# Predictive Analytics Project Report - Budhi Pamungkas

## Project Domain
Predictive maintenance is a critical application of machine learning in modern industrial systems. Unexpected machine failures can result in high operational costs, production downtime, and safety risks. Therefore, the ability to estimate machine failure risk before breakdown occurs is essential.

This project focuses on developing a predictive analytics model using the AI4I 2020 Predictive Maintenance dataset. The goal is to predict machine failure probability based on operational measurements such as temperature, torque, rotational speed, and tool wear.

Two ensemble learning algorithms, Random Forest and LightGBM, are implemented and compared to determine the best-performing model.

## Business Understanding

### Problem Statements
- How can machine operational features be used to predict machine failure?
- Which machine learning model performs better in failure detection: Random Forest or LightGBM?

### Goals
- Develop a predictive model to estimate the probability of machine failure.
- Compare Random Forest and LightGBM using appropriate evaluation metrics.

    ### Solution statements
    To achieve the objectives:
    1. Build two classification models:
        - Random Forest
        - LightGBM
    2. Evaluate models using:
        - Receiver Operating Characteristic - Area Under the Curve (ROC-AUC)
        - Precision-Recall - Area Under the Curve (PR-AUC)
        - Confusion Matrix
        - Precision, Recall, F1-score

## Data Understanding
The dataset used in this project is the AI4I 2020 Predictive Maintenance Dataset, available from:

[Kaggle](https://www.kaggle.com/datasets/chcarneiro/ai4i2020-csv)

This dataset contains synthetic industrial machine operational data and corresponding failure labels.

### Dataset Overview
- Total samples: approximately 10,000
- Target variable: Machine failure
- Class imbalance: Failure cases represent a small portion of the dataset (approximately 3.5%)

### Data Quality Assessment

To ensure the dataset meets the evaluation criteria under the **Data Understanding** section, several data quality checks were performed:

**Duplicate Data**
- Checked for duplicate records using `df.duplicated().sum()`.
- Result: No duplicate rows were found in the dataset.

**Missing Values**
- Checked for missing values using `df.isna().sum()`.
- Result: No missing values were detected in any feature.  

**Data Type Consistency**
- Verified data types using `df.info()`.
- All numerical features are stored in appropriate numeric formats.
- The categorical feature `Type` is correctly identified and encoded during preprocessing using ***one-hot encoding***.

### Variables in The AI4I 2020 Dataset
| Feature | Description |
|----------|------------|
| UDI | Unique data identifier |
| Product ID | Product serial number |
| Type | Product quality type (L, M, H) |
| Air temperature [K] | Ambient air temperature |
| Process temperature [K] | Internal process temperature |
| Rotational speed [rpm] | Machine rotational speed |
| Torque [Nm] | Applied torque |
| Tool wear [min] | Tool usage duration |
| Machine failure | Binary failure label (Target) |
| TWF | Tool Wear Failure |
| HDF | Heat Dissipation Failure |
| PWF | Power Failure |
| OSF | Overstrain Failure |
| RNF | Random Failure |

### Exploratory Data Analysis
The dataset is imbalanced. The target variable "Machine failure" shows that failure cases represent only a small proportion of the total observations (approximately 3.5%), while the majority of samples correspond to non-failure cases.

This imbalance is important because it can bias the model toward predicting the majority class (non-failure). As a result, accuracy alone would not be an appropriate evaluation metric.

## Data Preparation
The following preprocessing steps were performed:

1. Removal of Irrelevant Features
   - `UDI`, `Product ID`, and `Machine failure` were removed from the feature set using `df.drop(columns=...)`.
   - `UDI` and `Product ID` serve only as identifiers and do not contain predictive information, whilst `Machine failure` serve as target.
   - Including identifier features may introduce noise and reduce model generalization performance.

2. Target Variable Selection
   - The variable `Machine failure` was selected as the target variable.
   - This converts the problem into a supervised binary classification task.

3. Categorical Encoding
   - The categorical feature `Type` was encoded using **One-Hot Encoding**.
   - One-Hot Encoding was chosen because the categories (L, M, H) do not represent ordinal relationships, so the result would be ([1,0,0],[0,1,0],[0,0,1]).
   - This technique prevents the model from incorrectly assuming a numerical order between categories.

   ***One-hot Encoding***
   One-hot encoder is a data preprocessing technique that converts categorical data (e.g., "red," "green," "blue") into numerical binary vectors (e.g., [1,0,0],[0,1,0],[0,0,1]). It creates a new binary column for each unique category, marking 1 for presence and 0 for absence. This method prevents algorithms from assuming ordinal relationships.

4. Train-Test Splitting
   - The dataset was split into 80% training data and 20% testing data.
   - The `train_test_split()` function from scikit-learn was used.
   - Stratified sampling (`stratify=y`) was applied to preserve the original class distribution in both training and testing sets.
   - Stratification is important due to the imbalanced nature of the dataset (failure cases represent only a small proportion of the total observations (approximately 3.5%), while the majority of samples correspond to non-failure cases).

5. Handling Class Imbalance
   - Since failure cases represent only approximately 3.5% of the dataset, class imbalance handling was necessary.
   - For Random Forest, `class_weight="balanced"` was applied to penalize misclassification of the minority class.
   - For LightGBM, `scale_pos_weight` was used to adjust the importance of positive samples.
   - These techniques help improve the model's ability to detect failure cases.

## Modeling
Two ensemble models were developed: Random Forest and LightGBM.  
The parameters used in the notebook are explained below.

### 1. Random Forest
Random Forest is an ensemble learning algorithm that builds multiple decision trees and aggregates their results.

**Advantages:**
- Handles nonlinear relationships
- Robust to noise
- Provides feature importance

**Key Parameters:**
- **n_estimators = 500**  
  This parameter defines the number of decision trees in the forest. A higher number of trees generally improves model stability and reduces variance.
- **random_state = 260226**  
  This parameter ensures reproducibility of results. By fixing the random seed, the same dataset split and model behavior can be reproduced.
- **n_jobs = -1**  
  This parameter allows the model to use all available CPU cores during training. It improves computational efficiency and reduces training time.
- **class_weight = "balanced"**  
  Since the dataset is imbalanced (failure cases represent only a small proportion of the total observations (approximately 3.5%), while the majority of samples correspond to non-failure cases), this parameter automatically adjusts weights inversely proportional to class frequencies. It increases the importance of the minority class (`Machine failure` value on 1).

### 2. LightGBM
LightGBM is a gradient boosting framework optimized for efficiency and performance.

**Advantages:**
- Fast training
- Strong performance on structured datasets
- Handles imbalanced data effectively

**Key Parameters:**
- **n_estimators = 500**  
  Specifies the number of boosting iterations. More estimators allow the model to learn more complex patterns.
- **random_state = 260226**  
  This parameter ensures reproducibility of results. By fixing the random seed, the same dataset split and model behavior can be reproduced.
- **n_jobs = -1**
  This parameter allows the model to use all available CPU cores during training. It improves computational efficiency and reduces training time.
- **scale_pos_weight = (negative / positive ratio)**  
  This parameter adjusts the weight of the positive class to handle class imbalance (failure cases represent only a small proportion of the total observations (approximately 3.5%), while the majority of samples correspond to non-failure cases). It helps the model focus more on correctly predicting failure cases.
- **verbosity = -1**  
  Controls the level of training output messages. Setting it to -1 suppresses warning and informational messages during model training, making the output cleaner.

## Evaluation
The goals of this project were to develop a predictive model capable of detecting machine failure based on operational features and to compare the performance of Random Forest and LightGBM algorithm. 

Because the dataset is imbalanced (failure cases represent only a small proportion of the total observations (approximately 3.5%), while the majority of samples correspond to non-failure cases), the following metrics were used:
- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score
- Confusion Matrix

Both models are evaluated using threshold 0.5 . 
The models generate probability scores that represent the likelihood of machine failure for each observation. To convert these probabilities into class predictions, a decision threshold is applied.

With a threshold of 0.5:
- If the predicted probability >= 0.5, the instance is classified as **Failure (Class 1)**.
- If the predicted probability < 0.5, the instance is classified as **Non-Failure (Class 0)**.

The threshold of 0.5 is the default decision boundary in binary classification, as it represents equal likelihood between the two classes. All reported evaluation metrics, including the confusion matrix, precision, recall, and F1-score, were calculated based on this threshold.

Although 0.5 is commonly used as a standard threshold, different thresholds may be selected in real-world applications depending on the relative importance of false positives and false negatives.

### Model Performance Summary
| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-score | Confusion Matrix |
|--------|---------|---------|-----------|--------|----------|------------------|
| Random Forest | 0.9814 | 0.9716 | 1.00000 | 0.97059 | 0.98507 | [[1932, 0], [2, 66]] |
| LightGBM | 0.9943 | 0.9757 | 1.00000 | 0.97059 | 0.98507 | [[1932, 0], [2, 66]] |

#### ROC-AUC
ROC-AUC (Receiver Operating Characteristic - Area Under the Curve) measures the model's ability to distinguish between failure and non-failure cases across all possible classification thresholds. The ROC curve plots the True Positive Rate (Recall) against the False Positive Rate. A higher ROC-AUC value indicates better overall ranking performance.

In this project:
- Random Forest achieved a ROC-AUC of 0.9814.
- LightGBM achieved a higher ROC-AUC of 0.9943.

This indicates that LightGBM has a superior ability to rank failure cases higher than non-failure cases across different thresholds.

#### PR-AUC
PR-AUC (Precision-Recall Area Under the Curve) evaluates the trade-off between precision and recall across different classification thresholds. PR-AUC is particularly important for imbalanced datasets because it focuses on the performance of the positive (failure) class.

In this project:
- Random Forest achieved a PR-AUC of 0.9716.
- LightGBM achieved a PR-AUC of 0.9757.

The slightly higher PR-AUC of LightGBM indicates better performance in identifying failure cases while maintaining high precision.

#### Precision
Precision measures the proportion of predicted failure cases that are actually true failures.

Both models achieved a precision of 1.0000, meaning that every predicted failure case was correct. There were no false positive predictions (FP = 0).

This indicates that the model does not generate false alarms.

#### Recall
Recall (also called Sensitivity) measures the proportion of actual failure cases that were correctly identified.

Both models achieved a recall of 0.97059, meaning that approximately 97% of actual failure cases were successfully detected.

This indicates that the model has strong detection capability.

#### F1-score
The F1-score is the harmonic mean of precision and recall.

Both models achieved an F1-score of 0.98507, indicating a strong balance between precision and recall.

Since both precision and recall are high, the F1-score confirms that the models perform consistently well in detecting failure cases while minimizing errors.

#### Confusion Matrix
The confusion matrix for both Random Forest and LightGBM is:
\[
\begin{bmatrix}
1932 & 0 \\
2 & 66
\end{bmatrix}
\]

Where:
- True Negatives (TN) = 1932  
- False Positives (FP) = 0  
- False Negatives (FN) = 2  
- True Positives (TP) = 66  

This means:
- The model correctly classified 1932 non-failure cases.
- No false alarms were generated (FP = 0), resulting in perfect precision (1.0000).
- Only 2 failure cases were missed.
- 66 out of 68 failure cases were correctly detected, resulting in a recall of 0.9706.

These results indicate that the model performs exceptionally well in distinguishing between failure and non-failure cases while maintaining high detection capability for the minority class.

### Interpretation of Results
Both models achieved very high performance across all evaluation metrics. The recall value of 0.97059 indicates that approximately 97% of actual failure cases were correctly detected. This demonstrates that the models are highly effective at identifying machine failure events.

The precision score of 1.00000 indicates that all predicted failure cases were correct, meaning the model did not produce false positives at the default threshold of 0.5.

Although both models achieved identical precision, recall, F1-score, and confusion matrix at the classification threshold of 0.5, LightGBM demonstrated superior ranking capability, as shown by higher ROC-AUC and PR-AUC scores. This indicates that LightGBM is better at separating failure and non-failure cases across different thresholds.

### Generalization and Overfitting Consideration
Model evaluation was performed on unseen test data (20% split), which was not used during training. The high performance on the test set suggests that the models generalize well to new data.

There is no strong indication of overfitting, as the performance metrics remain consistently high without signs of instability. However, further validation using cross-validation could provide additional confirmation.

### Alignment with Business Goals
The primary goal of this project was to develop a model capable of predicting machine failure accurately. Based on the high recall and strong ROC-AUC and PR-AUC scores, the developed models successfully address the problem statement.

LightGBM is selected as the final model due to its superior ranking performance compared to Random Forest.

**---Ini adalah bagian akhir laporan---**