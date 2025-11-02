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
#### Random seed is for same result reproducibility. So, that at every run same result could be produced.
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
### Encoding Sentiment Labels
#### To make the labels suitable for training the model, it is converted into numerical form:
```
label_map = {-1.0: 0, 0.0: 1, 1.0: 2}
inv_label_map = {v: k for k, v in label_map.items()}
df["y"] = df["category"].map(label_map).astype(int)
```

### Handling Missing Data
#### For data quality, rows with missing values in the column of category are removed and it is done by using follwing command
```
df = df.dropna(subset=["category"]).copy()
```

### Text Cleaning and Preprocessing
#### I wrote a text cleaning function to prepare the tweets for modelling. The preprocessing included:
##### Converting all text to lowercase.
##### Removing URLs and user mentions like @username.
##### Keeping only alphabetic words and hashtags.
##### Replacing multiple spaces with a single one.
```
URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")

def basic_clean(text: str) -> str:
    text = str(text).lower().strip()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = re.sub(r"[^a-zA-Z#]+", " ", text)  # remove digits/punctuations
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text"] = df["clean_text"].astype(str).map(basic_clean)
df = df[["text", "y"]].dropna().reset_index(drop=True)

df.head()
```

<img width="578" height="268" alt="image" src="https://github.com/user-attachments/assets/36a6a93b-7940-498a-8b44-f5e05432544c" />

### Splitting the Dataset
#### Now, the dataset is split into training, validation and testing parts 
```
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    df["text"], df["y"],
    test_size=0.30, random_state=RANDOM_SEED, stratify=df["y"]
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=2/3, random_state=RANDOM_SEED, stratify=y_temp
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
```
<img width="422" height="48" alt="image" src="https://github.com/user-attachments/assets/ed770998-db58-46ec-be4a-37f19bb5dd49" />

### Building Baseline Model
#### For the baseline, I used the TF-IDF vectorizer along with a Multinomial Naive Bayes classifier.
```
baseline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1,2),
        min_df=5,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents="unicode"
    )),
    ("clf", MultinomialNB())
])
baseline.fit(X_train, y_train)
```

### Evaluating Baseline Model on validation data 
```
pred_val = baseline.predict(X_val)
print("\nBaseline Validation Results")
print("Accuracy:", accuracy_score(y_val, pred_val))
print("Macro-F1:", f1_score(y_val, pred_val, average="macro"))
print(classification_report(y_val, pred_val, digits=3))
```
<img width="601" height="282" alt="image" src="https://github.com/user-attachments/assets/43f397ca-f311-4b25-a3d1-a71a0af8b2b5" />

### Evaluating Baseline Model on test data
```
pred_test = baseline.predict(X_test)
print("\nBaseline Test Results")
print("Accuracy:", accuracy_score(y_test, pred_test))
print("Macro-F1:", f1_score(y_test, pred_test, average="macro"))
print(classification_report(y_test, pred_test, digits=3))
```
<img width="602" height="286" alt="image" src="https://github.com/user-attachments/assets/ff240b20-675e-4119-a599-d13c6f196518" />

```
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_test))
```
<img width="298" height="111" alt="image" src="https://github.com/user-attachments/assets/d15502da-18f6-410f-a604-2e427ec19bb2" />


### Preparing Data for Neural Model
#### This step converts text into numerical form using a vocabulary limit of 40,000 words and ensures all sequences are 64 tokens long. Unknown words are replaced with the <OOV> token, and class labels are transformed into one-hot vectors for the softmax layer.
```
VOCAB_SIZE = 40000
MAX_LEN = 64

# Tokenizer
tokenizer = keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

def vectorize(texts):
    seqs = tokenizer.texts_to_sequences(texts)
    return keras.preprocessing.sequence.pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

Xtr = vectorize(X_train)
Xva = vectorize(X_val)
Xte = vectorize(X_test)

# One-hot labels for softmax
lb = LabelBinarizer()
ytr = lb.fit_transform(y_train)
yva = lb.transform(y_val)
yte = lb.transform(y_test)
```
### Building BiLSTM Model
#### A BiLSTM model is built to understand patterns and context in text data.
#### It uses embedding and bidirectional LSTM layers to extract key text features.
#### Dropout layers are applied to prevent the model from overfitting.
#### Training is done with the Adam optimizer and categorical cross-entropy loss function.
#### Early stopping observes validation loss and ends training after three epochs without improvement, keeping the best model version.
```
# Build BiLSTM Model

def build_bilstm(vocab_size=VOCAB_SIZE, emb_dim=128, max_len=MAX_LEN, num_classes=3):
    inputs = keras.Input(shape=(max_len,), dtype="int32")
    x = layers.Embedding(vocab_size, emb_dim, input_length=max_len)(inputs)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_bilstm()


callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
]
```
### Training the BiLSTM Model
#### Now fitting the data into the BiLSTM model and training the model 
```
history = model.fit(
    Xtr, ytr,
    validation_data=(Xva, yva),
    epochs=10,
    batch_size=256,
    callbacks=callbacks,
    verbose=1
)
```
<img width="1272" height="337" alt="image" src="https://github.com/user-attachments/assets/3e37cefe-3066-4499-857d-0cb940fd56f3" />

### Evaluating the Neural Model
#### Finally, checking the performance of BiLSTEM model by classification report, Accuracy and f1-score
```
probs = model.predict(Xte)
y_pred = np.argmax(probs, axis=1)
y_true = y_test.values

print("\nBiLSTM Test Results")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Macro-F1:", f1_score(y_true, y_pred, average="macro"))
print(classification_report(y_true, y_pred, digits=3))
```
<img width="582" height="335" alt="image" src="https://github.com/user-attachments/assets/e99fedf5-98eb-423e-98cb-72e324f2793f" />

```
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
````
<img width="267" height="105" alt="image" src="https://github.com/user-attachments/assets/692d786e-4cb6-4771-b9bc-18fec0c22d3c" />

### Error Analysis

#### representing five misclassified examples
```
Build DataFrame for test set
test_df = pd.DataFrame({"text": X_test.values, "true": y_test.values, "pred": y_pred})
mis = test_df[test_df["true"] != test_df["pred"]].copy()
```
#### Add model confidence
```
test_df_probs = pd.DataFrame(probs, columns=[0, 1, 2])
test_df = test_df.reset_index(drop=True).join(test_df_probs)
mis = test_df[test_df["true"] != test_df["pred"]]
```
#### Sample only 5 misclassified examples
```
samples = mis.sample(n=min(5, len(mis)), random_state=RANDOM_SEED)
```
#### Print results
```
for i, row in samples.iterrows():
    txt = row["text"]
    
    t = int(row["true"])   
    p = int(row["pred"])   
    conf = row[int(row["pred"])]

    print("----")
    print("Text:", txt)
    print("True label:", t, " Pred label:", p, " Confidence:", f"{conf:.3f}")

    hint = []
    if "#" in txt or "http" in txt:
        hint.append("contains hashtags/urls")
    if any(ch in txt for ch in ["!", "?", ":-", ":)"]):
        hint.append("strong punctuation/emoticon")
    if len(txt.split()) <= 3:
        hint.append("very short")
    if len(hint) == 0:
        hint = ["ambiguous / sarcastic / domain-specific"]
    print("Quick hint:", ", ".join(hint))
    print()
```
<img width="1686" height="538" alt="image" src="https://github.com/user-attachments/assets/0825199d-acfb-460e-b141-35f91b91a1f6" />

### Comparison between Naive bayes and BiLSTEM classifier 

```
The BiLSTM model outperformed Naive Bayes by a large margin around 24% higher accuracy and 29% higher macro-F1.
This shows that neural models handle complex and context-dependent text much better than traditional methods.
•	The Naive Bayes model provided a good baseline but couldn’t fully understand sarcasm or context-dependent sentiments.
•	The BiLSTM model delivered excellent performance, understanding sentence flow and word relationships effectively.
•	Most errors were linked to sarcastic, factual, or politically nuanced tweets.
```












