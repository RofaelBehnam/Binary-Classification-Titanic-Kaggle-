import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder


def categorical_to_numeric(data):
    # convert categorical attributes to numerical attributes
    for column in data.columns.values:
        if 'object' == data[column].dtype:
            lb_make = LabelEncoder()
            data[column] = lb_make.fit_transform(data[column]).astype(np.float64)
    return data


# read training dataset
train = pd.read_csv('dataset/train.csv', sep=',')
# remove extra spaces at the beginning and the end of column titles
train = train.rename(columns=lambda x: x.strip())
test = pd.read_csv('dataset/test.csv', sep=',')
# remove extra spaces at the beginning and the end of column titles
test = test.rename(columns=lambda x: x.strip())
test_labels = pd.read_csv('dataset/gender_submission.csv', sep=',')
# remove extra spaces at the beginning and the end of column titles
test_labels = test_labels.rename(columns=lambda x: x.strip())
# separate training labels from features
train_labels = train.Survived

train = train.drop('Survived', axis=1)

# remove features that may be helpless
train = train.drop('PassengerId', axis=1)
test = test.drop('PassengerId', axis=1)

train = train.drop('Name', axis=1)
test = test.drop('Name', axis=1)

train = train.drop('Ticket', axis=1)
test = test.drop('Ticket', axis=1)

#################################
train = categorical_to_numeric(train)
test = categorical_to_numeric(test)

# # Initialize classifier
gaussian_NB = GaussianNB()

# Train classifier
train = np.array(train.values.tolist())
train_labels = np.array(train_labels.values.tolist())
model = gaussian_NB.fit(train, train_labels)
target_pred = model.predict(test)
print np.array_equal(target_pred, test_labels)
