import pandas as pd
import zipfile

# Both will work, but will affect downstream
import dtsc330.classifiers.reusable_classifier
dtsc330.classifiers.reusable_classifier.ReusableClassifier()

# from dtsc300 import reusable_classifier
# reusable_classifier.ReusableClassifier()

# In a script, no name = main if you don't want
# Can be ugly, don't add good notes

zf = zipfile.ZipFile('data/wine+quality.zip')
df = pd.read_csv(zf.open('winequality-white.csv'), sep = ';')
print(df)

# Train test split
labels = df['quality'] > 5
features = df.drop(columns = ['quality'])

rc = dtsc330.classifiers.reusable_classifier.ReusableClassifier()
rc.assess(features, labels)