import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics

#Load training data
#n=normal a=abnormal
trainingDataset = pd.read_csv('TrainingData.txt', names=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,"n/a"])
guidelinePrices= trainingDataset.iloc[:,0:24]#24 hours
isAbnormal = trainingDataset.loc[:,"n/a"].values

#separating the training data for testing
X_train, X_test, y_train, y_test = train_test_split(guidelinePrices, isAbnormal, test_size=0.2, random_state=42)  #8000 records are used for training and 1000 records are used for testing.
classifier = SVC(kernel='linear', random_state = 42)
classifier.fit(X_train, y_train)

#testing the remaining training data(2000)
y_pred = classifier.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))#checking the accuracy of the testing part by using the acccuracy_score from the  sklearn library.metrics class.

""" checking for confussion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
"""

testingDataset = pd.read_csv('TestingData.txt', names=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])

#using the predict function of the svc module to label test data as normal and abnormal, this data has classified as normal and abnormal by using the svm algorithm
testPrediction = classifier.predict(testingDataset)
testingDataset["n/a"] = testPrediction

#creating a result file
testingDataset.to_csv("TestingResults.txt", header=False, index=False)
