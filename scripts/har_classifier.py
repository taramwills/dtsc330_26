"""
Train and evaluate a sleep classifier using heart rate and accelerometer data.

This script:
- Loads HAR data using the provided reader
- Engineers acceleration magnitude
- Resamples data to fixed time intervals
- Splits by person for training/testing
- Trains a reusable classifier
- Evaluates prediction accuracy
"""

from dtsc330.readers import har
from dtsc330.classifiers import reusable_classifier
import pandas as pd
import numpy as np

"""
Load HAR data, build features/labels, and assess classifier performance.

This script follows the class requirement that all transformations occur after importing the reader and classifier.
"""
# Initialize HAR reader and load data for first 10 participants
har_data = har.HAR("data/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0", 10)

# Create a working copy of the combined dataframe
full_df = har_data.df.copy()

"""
Convert timestamp column to a TimedeltaIndex so that pandas resample() can be applied.
"""
full_df.index = pd.to_timedelta(full_df['timestamp'], unit = 's')

"""
Get unique participant identifiers so that features are computed separately for each person.
"""
people = pd.unique(full_df['person'])

"""
Lists to store per-person training and testing data.

Each element in these lists is a DataFrame (features) or Series (labels) corresponding to one participant.
"""
features, labels, test_features, test_labels = [], [], [], []

for person in people:
    """
    Process one participant at a time to avoid averaging
    across different individuals.
    """
    print(f'Computing person {person + 1}')

    # Subset data for current participant
    df = full_df.loc[full_df['person'] == person].copy()

    """
    Compute acceleration magnitude to summarize motion into a single measure.
    """
    df["acc_mag"] = np.sqrt(df['acc_x'] ** 2 +
                            df['acc_y'] ** 2 +
                            df['acc_z'] ** 2)

    # Ensure time index is sorted before resampling
    df = df.sort_index()

    """
    Resample data into 10-second intervals and select the first observation in each interval.
    """
    df = df.resample('60s').first().dropna()

    """
    Separate predictor variables (features) and target variable (sleep labels).
    """
    fs = df.drop(columns = ["is_sleep", "timestamp", "person", "acc_x", "acc_y", "acc_z"])
    ls = df["is_sleep"]
    
    """
    Use first participant as test set and remaining participants as training set.
    """
    if person < 1:
        test_features.append(fs)
        test_labels.append(ls)
    else:
        features.append(fs)
        labels.append(ls)

"""
Train classifier using combined training data.

Manual splitting is used instead of train_test_split to ensure participant-level separation.
"""
classifier = reusable_classifier.ReusableClassifier('xgboost')
classifier.train(pd.concat(features), pd.concat(labels))

"""
Generate predictions on held-out participant data.
"""
pred_labels = classifier.predict(pd.concat(test_features))
test_labels = pd.concat(test_labels)

"""
Compute classification accuracy manually by comparing
predicted labels to true labels.
"""
correct = np.sum(pred_labels.astype(int) == test_labels.to_numpy().astype(int))
total = len(pred_labels)

print("Correct:", correct, "/", total)
print("Accuracy:", correct / total)