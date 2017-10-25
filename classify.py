import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
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
# remove extra spaces at the beginning and the end of column titles
# separate training labels from features
IDs = test.PassengerId
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
train = np.array(train.values.tolist())
train_labels = np.array(train_labels.values.tolist())

classifiers = [GaussianNB(), MultinomialNB(), BernoulliNB(), LogisticRegression(), KNeighborsClassifier()]
classifiers_name = ['Gaussian_NB', 'Multinomial_NB', 'Bernoulli_NB', 'Logistic_Regression', 'K_Neighbors']
for classifier, name in zip(classifiers, classifiers_name):
    f = open('results/using_standard_' + name + '_classifier.txt', 'w')
    # print 'using ', name, ' classifier'
    # Train classifier
    model = classifier.fit(train, train_labels)
    test_labels = classifier.predict(test)
    result = [list(a) for a in zip(IDs, test_labels)]
    f.write('PassengerId, Survived\r\n')
    for i in range(len(result)):
        f.write(str(result[i][0]) + ", " + str(int(result[i][1])) + "\r\n")
    f.close()
    # print '#################################'
