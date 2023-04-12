from utils import mnist_reader
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.svm import SVC
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plot
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import cv2 
from scipy import ndimage
# read the data
X_train, y_train = mnist_reader.load_mnist('/content/data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('/content/data/fashion', kind='t10k')
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state = random.randint(1,10000000),test_size=0.33,  shuffle=True)


X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 784))
X_val_reshaped = np.reshape(X_val, (X_val.shape[0], 784))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 784))

# Normalize the features
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train_reshaped)
X_val_norm = scaler.transform(X_val_reshaped)
X_test_norm = scaler.transform(X_test_reshaped)


# ========================================================================
# =========================***** KNN  *****===============================
# ========================================================================

# Create an KNN classifier 
clf= KNeighborsClassifier(n_neighbors=3 , metric='manhattan')
clf.fit(X_train_norm, y_train)
# Test the classifier on the validation data to tune hyper parameter 
# choose the maximum k=3 with manhattan distance 
print("===***** KNN  *****=== ")

predict_set_val = clf.predict(X_val_norm)
val_acc = accuracy_score(y_val,predict_set_val)
print("Validation Accuracy: ", val_acc)
val_pre = precision_score(y_val,predict_set_val,average='macro')
print("Validation precision: ", val_pre)
val_recall = recall_score(y_val,predict_set_val,average='macro')
print("Validation recall: ", val_recall)
val_f1 = f1_score(y_val,predict_set_val,average='macro')
print("Validation f1: ", val_f1)

# Test the classifier on the testing data
predict_set_test = clf.predict(X_test_norm)
test_acc = accuracy_score(y_test,predict_set_test)
print("test Accuracy: ", test_acc)
test_pre = precision_score(y_test,predict_set_test,average='macro')
print("testn precision: ", test_pre)
test_recall = recall_score(y_test,predict_set_test,average='macro')
print("test recall: ", test_recall)
test_f1 = f1_score(y_test,predict_set_test,average='macro')
print("test f1: ", test_f1)

# Test the classifier on the training data
predict_set_train = clf.predict(X_train_norm)
train_acc = accuracy_score(y_train,predict_set_train)
print("train Accuracy: ", train_acc)
train_pre = precision_score(y_train,predict_set_train,average='macro')
print("train precision: ", train_pre)
train_recall = recall_score(y_train,predict_set_train,average='macro')
print("train recall: ", train_recall)
train_f1 = f1_score(y_train,predict_set_train,average='macro')
print("train f1: ", train_f1)


