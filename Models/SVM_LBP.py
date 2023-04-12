import random
from utils import mnist_reader
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import csv
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pandas as pd
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import normalize
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')


X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state = random.randint(1,10000000),test_size=0.33,  shuffle=True)

X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 784))
X_val_reshaped = np.reshape(X_val, (X_val.shape[0], 784))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 784))

# Normalize the features
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train_reshaped)
X_val_norm = scaler.transform(X_val_reshaped)
X_test_norm = scaler.transform(X_test_reshaped)


def lbp_transform(X):
    X_lbp = []
    for i in range(X.shape[0]):
        image = X[i].reshape(28, 28)
        lbp = local_binary_pattern(image, 12, 1, method='nri_uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 784))
        hist = hist.astype("float")
        hist = normalize(hist.reshape(1, -1))
        X_lbp.append(hist.flatten())
    return np.array(X_lbp)

X_train1 = lbp_transform(X_train)
X_test1 = lbp_transform(X_test)
X_val1=lbp_transform(X_val)
X_test=np.ndarray(shape=((X_test1.shape[0]),1567)).astype('float64')
for  i in range(X_test1.shape[0]):
  X_test[i]=np.concatenate((X_test_norm[i], X_test1[i]))
X_train=np.ndarray(shape=((X_train1.shape[0]),1567)).astype('float64')
for  i in range(X_train1.shape[0]):
  X_train[i]=np.concatenate((X_train_norm[i], X_train1[i]))
X_val=np.ndarray(shape=((X_val1.shape[0]),1567)).astype('float64')
for  i in range(X_val1.shape[0]):
  X_val[i]=np.concatenate((X_val_norm[i], X_val1[i]))
  
clf = SVC(kernel='rbf', C=13)
clf.fit(X_train, y_train)


predict_set_val = clf.predict(X_val)
val_acc = accuracy_score(y_val,predict_set_val)
print("Validation Accuracy: ", val_acc)
val_pre = precision_score(y_val,predict_set_val,average='macro')
print("Validation precision: ", val_pre)
val_recall = recall_score(y_val,predict_set_val,average='macro')
print("Validation recall: ", val_recall)
val_f1 = f1_score(y_val,predict_set_val,average='macro')
print("Validation f1: ", val_f1)


# Test the classifier on the testing data
predict_set_test = clf.predict(X_test)
test_acc = accuracy_score(y_test,predict_set_test)
print("test Accuracy: ", test_acc)
test_pre = precision_score(y_test,predict_set_test,average='macro')
print("testn precision: ", test_pre)
test_recall = recall_score(y_test,predict_set_test,average='macro')
print("test recall: ", test_recall)
test_f1 = f1_score(y_test,predict_set_test,average='macro')
print("test f1: ", test_f1)

# Test the classifier on the training data
predict_set_train = clf.predict(X_train)
train_acc = accuracy_score(y_train,predict_set_train)
print("train Accuracy: ", train_acc)
train_pre = precision_score(y_train,predict_set_train,average='macro')
print("train precision: ", train_pre)
train_recall = recall_score(y_train,predict_set_train,average='macro')
print("train recall: ", train_recall)
train_f1 = f1_score(y_train,predict_set_train,average='macro')
print("train f1: ", train_f1)


