# Three-Class-Sentiment-Classifier

### STEPS OF THREE CLASS SENTIMENT CLASSIFIER

1-Importing Required Libraries – Loaded all essential Python libraries for data processing, visualization, and model building.

2-Setting Random Seed and Data Path – Fixed random seeds for reproducibility and defined the dataset location.

3-Loading the Dataset – Read the Twitter dataset into a DataFrame using pandas.

4-Handling Missing Data – Removed rows where sentiment labels were missing.

5-Encoding Sentiment Labels – Converted sentiment categories (-1.0, 0.0, 1.0) into numerical values (0, 1, 2).

6-Text Cleaning and Preprocessing – Applied a custom cleaning function to lowercase text, remove links, mentions, and unwanted symbols.

7-Splitting the Dataset – Divided data into training, validation, and testing sets (70%, 10%, and 20%).

8-Building Baseline Model – Implemented a Naive Bayes classifier using TF-IDF features for initial performance comparison.

9-Evaluating Baseline Model – Calculated accuracy, F1-scores, and confusion matrix for the Naive Bayes model.

10-Preparing Data for Neural Model – Tokenized tweets, padded sequences, and one-hot encoded target labels.

11-Building BiLSTM Model – Designed a Bidirectional LSTM neural network using TensorFlow/Keras.

12-Training the BiLSTM Model – Trained the model using early stopping to avoid overfitting.

13-Evaluating the Neural Model – Tested the BiLSTM on unseen data and computed performance metrics.

14-Error Analysis – Displayed five misclassified tweets and analyzed possible reasons for the errors.

### Importing Required Libraries
```
import os, re, random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelBinarizer
```
### Setting Random Seed and Data Path

```
RANDOM_SEED = 24
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

DATA_PATH = "/content/k248015 Syed Muhammad Salman Rizvi - Twitter_Data.csv"
```
### Loading the Dataset
```
Reading the twitter dataset into the dataframe using pandas 
df = pd.read_csv(DATA_PATH)
```

### Handling Missing Data
```
df = df.dropna(subset=["category"]).copy()
```
