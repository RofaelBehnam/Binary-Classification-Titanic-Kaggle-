import pandas as pd
from sklearn.naive_bayes import GaussianNB

train = pd.read_csv('dataset/train.csv', sep=',')
test = pd.read_csv('dataset/test.csv', sep=',')
test_labels = pd.read_csv('dataset/gender_submission.csv', sep=',')
print train['Survived'].values
train_labels = train.Survived
# # Initialize classifier
gaussian_NB = GaussianNB()
del train.Survived
# Train classifier
model = gaussian_NB.fit(train, train_labels)
print model
