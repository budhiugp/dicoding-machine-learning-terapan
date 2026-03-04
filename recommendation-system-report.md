# Nintendo Switch Game Recommendation System - Budhi Pamungkas

## Project Domain
The Nintendo Switch platform offers a very large catalog of games across various genres, publishers, and categories. While this diversity is beneficial, it also creates a choice overload problem, where users struggle to identify games that match their preferences.

This project develops a content-based recommendation system for Nintendo Switch games using game metadata such as title, genre/tags, synopsis, publisher, ranking information, and price category.

The main output of the system is a Top-N list of recommended games that are most similar to a selected query game.

## Business Understanding

### Problem Statements
1. How can we generate Top-N relevant Nintendo Switch game recommendations using only game metadata, without user interaction data (ratings, clicks, purchases)?
2. Which content representation approach performs better for capturing game similarity:
    - TF-IDF (lexical sparse features), or
    - Sentence-BERT embeddings (dense semantic features)?

### Goals
The goals of this project are:
1. Build a content-based recommendation pipeline that:
    - Accepts a single game title as input.
    - Returns Top-N most similar games.
2. Implement and compare two different modeling approaches to determine which provides better recommendation quality using offline evaluation.

### Solution Approach
Since the available dataset does not contain user interaction data (such as user ratings history or user-item matrix), this project focuses on implementing a **Content-Based Filtering** approach. 

**Content-Based Filtering** recommends items based on the similarity between item features, such as genre, description, tags, and other metadata.

Two different approaches are implemented:
1. Model 1: TF-IDF + Cosine Similarity
    - Convert combined textual metadata into TF-IDF vectors.
    - Compute cosine similarity between games.
    - Rank games based on similarity scores.
2. Model 2: Sentence-BERT + Cosine Similarity
    - Convert combined textual metadata into semantic embeddings using all-MiniLM-L6-v2 pretrained model.
    - Compute cosine similarity between dense embeddings.
    - Rank games based on similarity scores.

These two approaches allow comparison between:
-  Keyword-based similarity (TF-IDF)
- Semantic similarity (SBERT)

## Data Understanding
The dataset used in this project is Nintendo Switch games Dataset, available from:

[Kaggle](https://www.kaggle.com/datasets/pedroaltobelli/nintendo-switch-games)

### Dataset Overview
- Total rows: 13,587
- Total columns: 18
- Target variable: None (unsupervised recommendation problem).
- Recommendation type: Content-Based Filtering using game metadata.

### Data Quality Assessment
To ensure the dataset meets the evaluation criteria under the **Data Understanding** section, several data quality checks were performed:

**Duplicate Data**
- Checked for duplicate records using `games_df.duplicated().sum()`.
- Result: No duplicate rows were found in the dataset.

**Missing Values**
- Checked for missing values using `games_df.isna().sum()`.
- Result: Missing values were found in the columns "price", "Players", "Developed by", "Series", and "Collection".
- Columns "Players", "Developed by", "Series", and "Collection" were removed due to a high proportion of missing values, which may reduce data reliability, using `games_df.drop(columns=to_drop)`.  
- Since only a small number of records had missing values in the "price" column, those rows were removed using `games_df_1.dropna(subset=['price'])`. 

**Data Type Consistency**
- Verified data types using `df.info()`.
- Converted price and rankScore to numeric format later.

### Dataset Variables
| Attribute | Description |
|------------|------------|
| gender | Main genre classification of the game (e.g., Adventure). |
| rank type | Ranking status label (e.g., ranked). |
| id | Unique numerical identifier for each game. |
| name | Title of the Nintendo Switch game. |
| rankScore | Numerical rating score assigned to the game. |
| sinopsis | Short description or overview of the game content. |
| tags | Detailed tags describing genre, themes, and gameplay elements. |
| rank | Current ranking position of the game (e.g., 16 / 4512). |
| price | Game price. |
| Category | Main game category or genre. |
| Players | Number of supported players. |
| Published by | Company that published the game. |
| Europe release | Official release date in Europe. |
| US release | Official release date in the United States. |
| Japan release | Official release date in Japan. |
| Developed by | Game development studio/company. |
| Series | Franchise or series the game belongs to. |
| Collection | Collection or bundle classification. |

## Data Preparation
Before modeling, the following preprocessing steps were performed:

1. Removal of Unused Columns
    - Columns "id", "Europe release", "Japan release" and "US release" were removed because they do not contribute semantic information relevant to content similarity.
    - The "gender" column was excluded since similar information is already captured more comprehensively in the "tags" and "Category" columns.
    - Drop unused/unimportant column using `games_df = df.drop(columns=to_drop)`
2. No Tags
    - Checked for any games with empty tags filled with "no tags" as value using `sum(games_df_2["tags"].apply(lambda x: isinstance(x, str) and "no tags" in x.lower()))`
    - Result: It shows there are 6,325 games that has no tags on its tags column.
    - Since the recommendation system relies heavily on textual metadata for similarity computation, entries without tags provide insufficient descriptive information. Therefore, these records were removed to ensure higher-quality feature representation.
    - Drop the rows using `games_df_2[~games_df_2["tags"].str.lower().str.contains("no tags", na=False)].reset_index(drop=True)`
3. Text Preparation and Combination
    - To prepare input for feature extraction, the following textual columns were combined:
        - name
        - tags
        - sinopsis
        - Category
        - Published by
    - All text was:
        - Converted to lowercase
        - Ensured to be string type
        - Concatenated into a single combined text feature
4. Parsed the "rank" column into:
    - "rank_pos" using `rank_parsed.apply(lambda x: x[0])`
    - "rank_total" using `rank_parsed.apply(lambda x: x[1])`
5. Created a normalized ranking score ("rank_norm") using `rank_norm = 1 - rank_pos/rank_total`.
    - Assign "rank_pos", "rank_total", and "rank_norm" as 0 (zero) if the data "rank" column is unranked.
    - Ensured all textual features were converted to string format before vectorization.
6. Parse price into price-bucket
    - Instead of using raw numerical price values (which are less suitable for cosine similarity on text-based features), categorical labels are appended to the combined text representation.
    | Price Range | Category |
    |-------------|----------|
    | Missing / Invalid | `unknown_price` |
    | 0 | `free` |
    | < 10 | `budget` |
    | 10 <= 30 | `mid` |
    | >= 30 | `premium` |
7. Text Preparation and Combination
    - To prepare input for feature extraction, the following textual columns were combined:
        - name
        - Category
        - tags
        - sinopsis
        - Published by
        - rank type
        - price_bucket
    - All text was:
        - Converted to lowercase
        - Ensured to be string type
        - Concatenated into a single combined text feature
8. Genre/Tag Parsing for Evaluation
    For offline evaluation purposes, the "tags" column was processed into structured tag sets.
    Tags were:
    - Cleaned
    - Split using separators
    - Converted into comparable sets
    These tag sets were later used to compute Precision@K with genre overlap proxy.

### Feature Extraction

After preprocessing and text combination, two different feature extraction techniques were applied to convert textual metadata into numerical representations for similarity computation.

#### TF-IDF Feature Extraction
The combined textual data was transformed into numerical features using TF-IDF.
Steps performed:
1. Initialize TF-IDF vectorizer.
2. Fit the vectorizer on the combined text.
3. Transform text into a sparse TF-IDF matrix.
This produces high-dimensional sparse vectors capturing keyword importance.
The TF-IDF matrix is used in Model 1 for cosine similarity computation.

#### Sentence-BERT Feature Extraction
The same combined text was encoded using the pretrained SBERT model:

`all-MiniLM-L6-v2`

Steps performed:
1. Load SentenceTransformer model.
2. Encode combined text into dense embedding vectors.
3. Store embedding matrix for similarity computation.

This produces dense 384-dimensional semantic vectors.

The embedding matrix is used in Model 2 for cosine similarity computation.

## Modeling

This project implements two content-based recommendation models to generate Top-N similar Nintendo Switch games based on textual metadata similarity. Since the problem is unsupervised and no user interaction data is available, similarity is computed directly between item representations.

## Model 1: TF-IDF + Cosine Similarity

### Model Definition
TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical text representation technique that converts documents into high-dimensional sparse vectors based on word importance.

**Key Parameters:**
- **ngram_range = (1,2)**  
  This parameter defines the range of n-grams used during feature extraction. The value `(1,2)` means the model considers both single words (unigrams) and two-word phrases (bigrams). This allows the system to capture meaningful term combinations such as "action RPG" or "open world", which provide richer contextual information than individual words alone.
- **stop_words = "english"**  
  This parameter removes common English words that carry little semantic meaning, such as "the", "and", or "is". Eliminating these words helps reduce noise in the text representation and improves the quality of the extracted features.
- **min_df = 2**  
  This parameter specifies the minimum number of documents in which a term must appear to be included in the vocabulary. By setting `min_df = 2`, terms that appear only once in the dataset are removed. This helps eliminate extremely rare terms that may represent noise or typographical variations.
- **max_df = 0.9**  
  This parameter sets the maximum document frequency threshold. Terms that appear in more than 90% of the documents are removed because they are too common and do not help distinguish between different games.

### Cosine Similarity Computation (TF-IDF Model)
In the TF-IDF based recommendation model, cosine similarity is used to measure the similarity between the TF-IDF representation of a selected game and all other games in the dataset.

Each game is represented as a TF-IDF vector derived from the combined textual metadata (such as name, tags, category, synopsis, and publisher). These vectors capture the importance of terms within each game description.

The similarity between games is computed using 
`cosine_similarity(X_tfidf[idx], X_tfidf).ravel()`

Where:
- X_tfidf
  This matrix contains the TF-IDF vectors for all games in the dataset. Each row represents a game and each column represents a term from the vocabulary generated by the TF-IDF vectorizer. Assigned from code `tfidf.fit_transform(games_df_4["text"])`
- idx
  This variable represents the index of the query game selected by the user. It is obtained from the name_to_idx dictionary, which maps each game title to its corresponding row index in the dataset.
- cosine_similarity(X_tfidf[idx], X_tfidf)
  This function computes the cosine similarity between the TF-IDF vector of the selected game and the TF-IDF vectors of all other games. The result is a similarity score for each game in the dataset.
- ravel()
  This method converts the resulting similarity matrix into a one-dimensional array, making it easier to process and rank similarity scores.

### Model Characteristics
- Captures keyword-level similarity
- Highly sensitive to exact word overlap
- Produces sparse, high-dimensional vectors
- Computationally efficient

## Model 2: Sentence-BERT (SBERT) + Cosine Similarity

### Model Definition
Sentence-BERT (SBERT) is a transformer-based model designed to generate dense semantic embeddings for sentences. Unlike TF-IDF, SBERT captures contextual meaning and semantic relationships rather than relying solely on exact word frequency.

**Key Parameters:**
- **model_name = "all-MiniLM-L6-v2"**  
  This parameter specifies the pretrained Sentence-BERT model used to generate text embeddings. The `all-MiniLM-L6-v2` model is a lightweight transformer architecture that produces 384-dimensional sentence embeddings while maintaining good performance for semantic similarity tasks.

    *** About all-MiniLM-L6-v2 ***
    - **MiniLM**: Lightweight transformer architecture  
    - **L6**: 6 transformer layers  
    - **v2**: Improved second version  
    - **all-**: Trained on diverse datasets for general-purpose sentence embeddings  

- **normalize_embeddings = True**  
  This parameter ensures that the generated embedding vectors are L2-normalized. Normalization makes cosine similarity computation more stable and efficient, as each embedding vector has a unit length.
- **show_progress_bar = True**  
  This parameter enables a progress bar during the embedding generation process. It provides visual feedback when processing large datasets and helps monitor the encoding progress.

### Cosine Similarity Computation (SBERT Model)
In the Sentence-BERT based recommendation model, cosine similarity is used to measure the semantic similarity between the embedding representation of a selected game and all other games in the dataset.

Each game is represented as a dense embedding vector generated from the combined textual metadata. These embeddings capture the semantic meaning of the text rather than relying solely on exact word overlap.

The similarity between games is computed using:
`sims = embeddings @ embeddings[idx]`

Where:
- **embeddings**  
  This matrix contains the SBERT embedding vectors for all games in the dataset. Each row represents a game and each column corresponds to one dimension of the semantic embedding generated by the pretrained model `all-MiniLM-L6-v2`.
- **idx**  
  This variable represents the index of the query game selected by the user. It is obtained from the `name_to_idx_sbert` dictionary, which maps each game title to its corresponding row index in the dataset.
- **embeddings @ embeddings[idx]**  
  This operation performs a dot product between the embedding vector of the selected game and all other embedding vectors in the dataset.

Since the embeddings were generated using `normalize_embeddings=True`, each embedding vector has unit length. Under this condition, the dot product between two vectors is mathematically equivalent to cosine similarity.

### Model Characteristics
- Captures semantic similarity
- Recognizes contextual relationships and synonyms
- Produces dense, low-dimensional representations
- More computationally intensive than TF-IDF

## Recommendation Pipeline
For both models, the recommendation system follows the same general pipeline:
1. **Input Query**  
   The user provides the title of a Nintendo Switch game.
2. **Index Retrieval**  
   The system locates the corresponding index of the game in the dataset using a mapping dictionary.
3. **Similarity Computation**  
   The similarity between the selected game and all other games is computed using cosine similarity:
   - TF-IDF vectors for the TF-IDF model
   - Sentence-BERT embeddings for the SBERT model
4. **Ranking**  
   All games are ranked based on their similarity scores in descending order.
5. **Top-N Recommendation**  
   The system returns the Top-N games with the highest similarity scores as recommendations.

## Model Comparison Objective
The two models represent different similarity paradigms:
- **Lexical Similarity (TF-IDF)**  
  Measures similarity based on shared words and term frequency patterns.
- **Semantic Similarity (SBERT)**  
  Measures similarity based on contextual meaning and sentence-level representation.

The comparison focuses on:
- Relevance of Top-N recommendations
- Consistency of similarity ranking
- Ability to capture deeper semantic relationships between games

## Case Study: Query "Xenoblade Chronicles 3"
To further analyze model behavior, recommendations were generated for:

**Query Game:** Xenoblade Chronicles 3

### TF-IDF Top-10 Recommendations
| Rank | Game Title | Category | Publisher | Similarity Score |
|------|------------|----------|-----------|-----------------|
| 1 | Xenoblade Chronicles 2 | RPG > Action RPG | Nintendo | 0.853980 |
| 2 | Xenoblade Chronicles: Definitive Edition | RPG > Action RPG | Nintendo | 0.734297 |
| 3 | .hack//G.U. Last Recode | RPG > Action RPG | Bandai Namco Entertainment | 0.363891 |
| 4 | Ravensword: Shadowlands | RPG > Action RPG | Ratalaika Games | 0.340748 |
| 5 | Undungeon | RPG > Action RPG | tinyBuild Games | 0.310833 |
| 6 | Hatchwell | RPG > Action RPG | Adrian Corpuz | 0.306814 |
| 7 | Dragon's Dogma: Dark Arisen | RPG > Action RPG | Capcom | 0.305180 |
| 8 | Eternal Radiance | RPG > Action RPG | Visualnoveler | 0.304833 |
| 9 | The Elder Scrolls V: Skyrim | RPG > Action RPG | Bethesda Softworks | 0.302385 |
| 10 | Crystar | RPG > Action RPG | NIS America | 0.299045 |

Observation:
- The recommendations are strongly concentrated in the **RPG > Action RPG** category.
- Most recommended games share highly similar tags such as *fantasy*, *real-time combat*, and *third-person gameplay*.
- This indicates that the TF-IDF model prioritizes **keyword overlap and genre similarity** in the textual metadata.

### SBERT Top-10 Recommendations
| Rank | Game Title | Category | Publisher | Similarity Score |
|------|------------|----------|-----------|-----------------|
| 1 | Xenoblade Chronicles 2 | RPG > Action RPG | Nintendo | 0.949090 |
| 2 | Xenoblade Chronicles: Definitive Edition | RPG > Action RPG | Nintendo | 0.861304 |
| 3 | Dragon Ball Xenoverse 2 | RPG > Action RPG | Bandai Namco Entertainment | 0.771186 |
| 4 | Metal Max Xeno Reborn | RPG | PQube | 0.747950 |
| 5 | Crystar | RPG > Action RPG | NIS America | 0.656374 |
| 6 | Warframe | Shooting > Third-person shooter | Digital Extremes | 0.648215 |
| 7 | Fire Emblem Warriors: Three Hopes | RPG > Action RPG | Nintendo | 0.643918 |
| 8 | Daemon X Machina | Shooting | Nintendo | 0.629173 |
| 9 | Minecraft Legends | RPG > Action RPG | Mojang | 0.625568 |
| 10 | Earthen Dragon | RPG > Action RPG | Origamihiro Games | 0.621905 |

Observation:
- The recommendations include games from multiple related genres beyond pure Action RPG.
- Several titles share thematic or stylistic similarities rather than identical tags.
- This suggests that the SBERT model captures **semantic relationships** between game descriptions rather than relying solely on keyword overlap.

### Comparative Analysis
**TF-IDF Model**
- Produces tightly clustered recommendations within the same genre category.
- Prioritizes games with strong keyword overlap in textual metadata.
- Achieves higher Precision@10 under the genre-overlap evaluation metric.

**SBERT Model**
- Produces more diverse recommendations across related genres.
- Captures semantic similarity between game descriptions even when keywords differ.
- Slightly lower Precision@10 under the genre-overlap metric.

The difference suggests that the evaluation metric favors lexical similarity. Since Precision@10 is based on tag overlap, models relying on keyword matching (such as TF-IDF) naturally achieve higher scores. In contrast, SBERT may recommend semantically related games that do not share identical tags, which can reduce the measured score despite producing meaningful recommendations.

## Evaluation
Since the dataset does not include user interaction data (ratings, clicks, or purchase history), evaluation was conducted using an offline proxy metric based on genre/tag overlap.

### Offline Evaluation: Precision@10 (Genre Overlap Proxy)
The metric was computed across multiple query games in the dataset, and the average Precision@10 score was used as the final evaluation result. 

A recommendation is considered relevant if it shares at least one genre or tag with the query game. Therefore, games labeled as "no tags" were removed during data preparation, since tags are an important feature for computing recommendation relevance in this project.

$$
Precision@10 =
\frac{\text{Number of Relevant Games in Top-10}}{10}
$$

### Quantitative Evaluation Results
| Model | Precision@10 |
|--------|--------------|
| TF-IDF + Cosine Similarity | 0.9057 |
| SBERT + Cosine Similarity | 0.8427 |

### Interpretation of Results

#### 1. TF-IDF Performance (0.9057)
TF-IDF achieved a higher Precision@10 score. This indicates that:
- TF-IDF performs very well when relevance is defined strictly by tag overlap.
- Since TF-IDF relies on exact keyword matching, it strongly favors games sharing identical genre labels.
- The evaluation metric (genre overlap) aligns directly with TF-IDF's lexical nature.

TF-IDF benefits from the evaluation definition because both operate at the word level.

#### 2. SBERT Performance (0.8427)
SBERT achieved a slightly lower Precision@10 score. Possible explanation:
- SBERT captures semantic similarity rather than exact keyword overlap.
- It may recommend games that are conceptually similar but do not share identical tags.
- Because the evaluation metric only counts tag overlap, semantically related games with different labels may be penalized.

SBERT may generate meaningful recommendations that are not fully captured by this proxy metric.

## Final Evaluation Conclusion
Based on Precision@10:
- TF-IDF achieves higher performance under the defined genre-overlap-based Precision@10 metric.
- SBERT provides competitive performance while offering deeper semantic understanding.

If the system objective is:
- Strict genre consistency > TF-IDF is preferable.
- Semantic richness and conceptual similarity > SBERT is more flexible.

Future improvements could include:
- Human-based relevance evaluation
- Hybrid similarity (combining TF-IDF and SBERT)
- Incorporating ranking normalization features
- User interaction-based validation

**---Ini adalah bagian akhir laporan---**