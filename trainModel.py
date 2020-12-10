import tensorflow as tf
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tflearn



import numpy as np
import cv2




train_data = np.load('training_data.npy',allow_pickle=True)
test_data = np.load('testing_data.npy',allow_pickle=True)
#LEARNİNG RATE
LR =0.001





convnet = input_data(shape=[None, 28,28, 1], name='input')

convnet = conv_2d(convnet,32,3,activation='relu');
convnet = max_pool_2d(convnet,3)

convnet = conv_2d(convnet,32,3,activation='relu');
convnet = max_pool_2d(convnet,3)

convnet = conv_2d(convnet,32,3,activation='relu');
convnet = max_pool_2d(convnet,3)


convnet = conv_2d(convnet,32,3,activation='relu');
convnet = max_pool_2d(convnet,3)


convnet = fully_connected(convnet, 256, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


X = []
for x in range(0,len(train_data)):
	X.append(train_data[x][0])

X = np.array(X).reshape(-1,28,28,1)
print(X.shape)

Y = []

for x in range(0,len(train_data)):
	temp = np.zeros(10)
	temp[train_data[x][1]] =1;
	Y.append(temp)



test_X =  []

for x in range(0,len(test_data)):
	test_X.append(test_data[x][0])

test_X = np.array(test_X).reshape(-1,28,28,1)	

test_Y = []


for x in range(0,len(test_data)):
	temp = np.zeros(10)
	temp[test_data[x][1]] = 1
	test_Y.append(temp)	







model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_X}, {'targets': test_Y}), 
    snapshot_step=500, show_metric=True, run_id="Test")


model.save("Testmodel")