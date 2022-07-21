import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

penguins = pd.read_csv('Penguins.csv')

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = penguins.copy()
target = 'species'  # predict the species of the penguins
encode = ['sex', 'island']

# encoding ['sex', 'island'] columns into ['male', 'female'...] columns
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

# print(df)

target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}


def target_encode(val):
    """
        Encode:
            'Adelie': 0
            'Chinstrap': 1
            'Gentoo': 2
    """

    return target_mapper[val]


df['species'] = df['species'].apply(target_encode)

# Separating:
#   X = indepenment var
#   Y = dependent var
X = df.drop('species', axis=1)  # drop column
Y = df['species']

# Build Random Forest model
clf = RandomForestClassifier()
clf.fit(X, Y)  # train model

# Saving the model
pickle.dump(clf, open('penguins_clf.pkl', 'wb'))
