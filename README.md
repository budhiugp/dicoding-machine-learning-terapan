# Dicoding – Machine Learning Terapan

This repository contains machine learning projects developed during the **Dicoding: Machine Learning Terapan** course.

The projects demonstrate practical machine learning applications for:
- **Predictive maintenance in industrial systems**
- **Content-based recommendation systems**

---


Detailed project reports are provided separately for clarity and documentation.

---

# Projects

## 1. Predictive Maintenance Analytics

### Overview
Predictive maintenance is an important application of machine learning in industrial environments. Unexpected machine failures may cause production downtime, increased operational costs, and safety risks.

This project develops a **predictive analytics model** that estimates the probability of machine failure based on operational measurements.

### Dataset
AI4I 2020 Predictive Maintenance Dataset  
https://www.kaggle.com/datasets/chcarneiro/ai4i2020-csv

Dataset characteristics:
- ~10,000 observations
- Industrial operational measurements
- Binary classification problem
- Imbalanced dataset (~3.5% failure cases)

### Features
The model uses machine operational parameters such as:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Product type

### Models Implemented

Two ensemble models were implemented and compared:

**Random Forest**
- Robust ensemble tree model
- Handles nonlinear relationships well

**LightGBM**
- Gradient boosting framework optimized for speed and performance

### Evaluation Metrics

Due to class imbalance, multiple evaluation metrics were used:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score
- Confusion Matrix

### Results

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-score |
|------|------|------|------|------|------|
| Random Forest | 0.9814 | 0.9716 | 1.000 | 0.9706 | 0.9851 |
| LightGBM | **0.9943** | **0.9757** | 1.000 | 0.9706 | 0.9851 |

LightGBM achieved the best ranking performance.

### Project Outcome
The model demonstrates strong performance in detecting machine failure events and shows potential application in **predictive maintenance systems for industrial operations**.

Full project report:  
`predictive-maintenance-report.md`

---

## 2. Nintendo Switch Game Recommendation System

### Overview
Modern gaming platforms offer thousands of available titles. This large catalog often creates a **choice overload problem**, where users struggle to identify games that match their preferences.

This project develops a **content-based recommendation system** for Nintendo Switch games using game metadata.

The system produces **Top-N game recommendations** based on similarity with a selected query game.

### Dataset
Nintendo Switch Games Dataset  
https://www.kaggle.com/datasets/pedroaltobelli/nintendo-switch-games

Dataset characteristics:
- 13,587 games
- 18 attributes
- Metadata including genre, synopsis, tags, publisher, and price.

### Feature Engineering

Textual metadata from several columns were combined:

- game title
- category
- tags
- synopsis
- publisher
- price category
- ranking information

Text preprocessing included:
- normalization
- feature combination
- tag parsing for evaluation

### Models Implemented

Two recommendation approaches were compared.

#### TF-IDF + Cosine Similarity
- Sparse lexical feature representation
- Captures keyword-based similarity
- Computationally efficient

#### Sentence-BERT + Cosine Similarity
- Dense semantic embeddings
- Captures contextual similarity
- Uses pretrained model `all-MiniLM-L6-v2`

### Evaluation

Offline evaluation used **Precision@10** based on genre/tag overlap.

| Model | Precision@10 |
|------|------|
| TF-IDF + Cosine Similarity | **0.9057** |
| SBERT + Cosine Similarity | 0.8427 |

### Key Findings

TF-IDF achieved higher Precision@10 because the evaluation metric favors lexical overlap.

However, SBERT captures **semantic relationships between games**, producing more diverse and contextually relevant recommendations.

Full project report:  
`recommendation-system-report.md`

---

# Technologies Used

- Python
- Scikit-learn
- LightGBM
- SentenceTransformers
- Pandas
- NumPy
- Jupyter Notebook

---

# Learning Outcomes

Through these projects, the following machine learning concepts were applied:

- Data preprocessing and feature engineering
- Model training and evaluation
- Handling imbalanced datasets
- Ensemble learning methods
- Text feature extraction (TF-IDF)
- Semantic embeddings using Sentence-BERT
- Content-based recommendation systems

---

# Future Improvements

Potential future improvements include:

- Hybrid recommendation models
- Human-based evaluation of recommendations
- Integration of user interaction data
- Deployment of predictive maintenance models in industrial monitoring systems

---

# Author

Budhi Pamungkas

Machine Learning projects developed as part of the Dicoding learning path.
