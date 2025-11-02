# Three-Class-Sentiment-Classifier

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
