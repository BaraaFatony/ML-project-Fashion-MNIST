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

roberts_cross_v = np.array( [[-1,0,1],
                             [-1,0,1],
                              [-1,0,1]] )
  
roberts_cross_h = np.array( [[-1,-1,-1],
                             [0,0,0],
                              [1,1,1]] )
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
# =========================***** Edge-train  *****========================
# ========================================================================

edged_img_X_train=np.ndarray(shape=((X_train.shape[0]),28,28)).astype('uint8')
reshaped_img_X_train=np.ndarray(shape=((X_train.shape[0]),28,28)).astype('uint8')

for  i in range(X_train.shape[0]):
  reshaped_img=np.zeros(shape=(28,28)).astype('uint8')
  reshaped_img=X_train[i].reshape(28,28)
  reshaped_img_X_train[i]=reshaped_img
preedged_img_X_train=np.ndarray(shape=((X_train.shape[0]),28,28)).astype('float64')

for  i in range(X_train.shape[0]):
  preedged_img_X_train[i]=reshaped_img_X_train[i]/255.0
  vertical_X_train = ndimage.convolve( preedged_img_X_train[i], roberts_cross_v )
  horizontal_X_train = ndimage.convolve( preedged_img_X_train[i], roberts_cross_h )
  
  edged_image = np.sqrt( np.square(horizontal_X_train) + np.square(vertical_X_train))
  edged_image=edged_image*255
  edged_img_X_train[i]=edged_image
  
# ========================================================================
# =========================***** Edge-val  *****==========================
# ========================================================================

edged_img_X_val=np.ndarray(shape=((X_val.shape[0]),28,28)).astype('uint8')
reshaped_img_X_val=np.ndarray(shape=((X_val.shape[0]),28,28)).astype('uint8')

for  i in range(X_val.shape[0]):
  reshaped_img=np.zeros(shape=(28,28)).astype('uint8')
  reshaped_img=X_val[i].reshape(28,28)
  reshaped_img_X_val[i]=reshaped_img
preedged_img_X_val=np.ndarray(shape=((X_val.shape[0]),28,28)).astype('float64')

for  i in range(X_val.shape[0]):
  preedged_img_X_val[i]=reshaped_img_X_val[i]/255.0
  vertical_X_val = ndimage.convolve( preedged_img_X_val[i], roberts_cross_v )
  horizontal_X_val = ndimage.convolve( preedged_img_X_val[i], roberts_cross_h )
  
  edged_image = np.sqrt( np.square(horizontal_X_val) + np.square(vertical_X_val))
  edged_image=edged_image*255
  edged_img_X_val[i]=edged_image
  
# ========================================================================
# =========================***** Edge-test  *****=========================
# ========================================================================

edged_img_X_test=np.ndarray(shape=((X_test.shape[0]),28,28)).astype('uint8')
reshaped_img_X_test=np.ndarray(shape=((X_test.shape[0]),28,28)).astype('uint8')

for  i in range(X_test.shape[0]):
  reshaped_img=np.zeros(shape=(28,28)).astype('uint8')
  reshaped_img=X_test[i].reshape(28,28)
  reshaped_img_X_test[i]=reshaped_img
preedged_img_X_test=np.ndarray(shape=((X_test.shape[0]),28,28)).astype('float64')

for  i in range(X_test.shape[0]):
  preedged_img_X_test[i]=reshaped_img_X_test[i]/255.0
  vertical_X_test = ndimage.convolve( preedged_img_X_test[i], roberts_cross_v )
  horizontal_X_test = ndimage.convolve( preedged_img_X_test[i], roberts_cross_h )
  
  edged_image = np.sqrt( np.square(horizontal_X_test) + np.square(vertical_X_test))
  edged_image=edged_image*255
  edged_img_X_test[i]=edged_image
 
# ========================================================================
# =========================***** Edge-reshape  *****======================
# ========================================================================

  
EX_train_reshaped = np.reshape(edged_img_X_train, (X_train.shape[0], 784))
EX_val_reshaped = np.reshape(edged_img_X_val, (X_val.shape[0], 784))
EX_test_reshaped = np.reshape(edged_img_X_test, (X_test.shape[0], 784))


# ========================================================================
# =========================***** Edge-normalize  *****====================
# ========================================================================

# Normalize the features
scaler = MinMaxScaler()
EX_train_norm = scaler.fit_transform(EX_train_reshaped)
EX_val_norm = scaler.transform(EX_val_reshaped)
EX_test_norm = scaler.transform(EX_test_reshaped)

# ========================================================================
# =========================***** concatenate-train  *****=================
# ========================================================================
X_train_final=np.ndarray(shape=((X_train.shape[0]),784*2)).astype('float64')
for  i in range(X_train.shape[0]):
  X_train_final[i]=np.concatenate((X_train_norm[i], EX_train_norm[i]))

  
# ========================================================================
# =========================***** concatenate-test  *****==================
# ========================================================================
X_test_final=np.ndarray(shape=((X_test.shape[0]),784*2)).astype('float64')
for  i in range(X_test.shape[0]):
  X_test_final[i]=np.concatenate((X_test_norm[i], EX_test_norm[i]))

# ========================================================================
# =========================***** concatenate-val  *****===================
# ========================================================================
X_val_final=np.ndarray(shape=((X_val.shape[0]),784*2)).astype('float64')
for  i in range(X_val.shape[0]):
  X_val_final[i]=np.concatenate((X_val_norm[i], EX_val_norm[i])) 




# Create an KNN classifier 
clf = MLPClassifier(hidden_layer_sizes=300,alpha=0.001, max_iter=300)
clf.fit(X_train_final, y_train)
# Test the classifier on the validation data to tune hyper parameter 
print("===***** MLP_Edge  *****=== ")

predict_set_val = clf.predict(X_val_final)
val_acc = accuracy_score(y_val,predict_set_val)
print("Validation Accuracy: ", val_acc)
val_pre = precision_score(y_val,predict_set_val,average='macro')
print("Validation precision: ", val_pre)
val_recall = recall_score(y_val,predict_set_val,average='macro')
print("Validation recall: ", val_recall)
val_f1 = f1_score(y_val,predict_set_val,average='macro')
print("Validation f1: ", val_f1)

# Test the classifier on the testing data
predict_set_test = clf.predict(X_test_final)
test_acc = accuracy_score(y_test,predict_set_test)
print("test Accuracy: ", test_acc)
test_pre = precision_score(y_test,predict_set_test,average='macro')
print("testn precision: ", test_pre)
test_recall = recall_score(y_test,predict_set_test,average='macro')
print("test recall: ", test_recall)
test_f1 = f1_score(y_test,predict_set_test,average='macro')
print("test f1: ", test_f1)


# Test the classifier on the training data
predict_set_train = clf.predict(X_train_final)
train_acc = accuracy_score(y_train,predict_set_train)
print("train Accuracy: ", train_acc)
train_pre = precision_score(y_train,predict_set_train,average='macro')
print("train precision: ", train_pre)
train_recall = recall_score(y_train,predict_set_train,average='macro')
print("train recall: ", train_recall)
train_f1 = f1_score(y_train,predict_set_train,average='macro')
print("train f1: ", train_f1)

