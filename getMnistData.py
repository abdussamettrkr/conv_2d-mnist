import tensorflow as tf
import numpy as np
import cv2



(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data();


train_data = []


for x in range(0,len(x_train)):	
	train_data.append([np.array(x_train[x]),np.array(y_train)[x]])


test_data=[]

for x in range(0,len(x_test)):
	test_data.append([np.array(x_test[x]),np.array(y_test)[x]])


np.save('training_data.npy',train_data);
np.save('testing_data.npy',test_data);