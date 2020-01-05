from srsmp import *
#from Preprocessings import pre_process, CalcRefScale

import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import json
import warnings
import configparser
import re

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import numpy.ma as ma
import keras
from keras.utils import to_categorical
from keras.layers import Dense, BatchNormalization, Activation, Dropout
from keras.layers import AveragePooling1D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint




#%% Definitions
# what to train
separate_data_sets = False
ML_baselines = False
basic_model = True
resnet_model = True

# data preprocessing
sel_WL = True
range_low = 500 # [nm]
range_high = 1000 # [nm]
use_t_int = False
data_augm_train = True
data_augm_test = False # normally set to False
preprocessing_method = 2

# NN general
train_size = 0.8
batch_size = 32
epochs = 2
val_split = 0.2
dropout_rate = 0.4
use_callbacks = True

# NN - ResNet
num_filters = 100



#%% load configuration and create log-file
# data paths from config.ini file
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read('config.ini')
check_and_init_all_dir(config)

train_path = config['NeuralNetTrain']['train_dir']
if separate_data_sets:
	test_path = config['NeuralNetTrain']['test_dir']

results_dir = config['NeuralNetTrain']['results_dir']

# class lookup from config.ini file
class_lookup = {}
for i in range(len(config['ClassLookup'])):
	class_lookup[config['ClassLookup'][str(i)]] = i

# create new file to log results
cur_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
cur_date = cur_datetime[:8]
log1 = open(results_dir + '/output/%s_log.txt' % (cur_datetime), 'w+')

# log current date and time
log1.write(cur_datetime + "\n\n")



#%% Get data
# get data from single source
def get_data(train_path,sel_WL,use_t_int,data_augm_train,train_size,preprocessing_method,class_lookup):
	# load from file
	X1,y1,labels,class_lookup = getData_binary(train_path,
											sel_WL=sel_WL,
											use_t_int=use_t_int,
											data_augm=data_augm_train,
											method=preprocessing_method,
											class_lookup=class_lookup)
	# transform 
	y1 = np.expand_dims(y1,axis=1)
	labels = np.expand_dims(labels,axis=1)
	y_l = np.concatenate((y1,np.asarray(labels)),axis=1)
	# split into training and test data
	X_train, X_test, y_l_train, y_l_test = train_test_split(X1,y_l,train_size=train_size, shuffle=True)
	y_train = y_l_train[:,0]
	y_test = y_l_test[:,0]
	labels_test = y_l_test[:,1]
	# shuffle to randomize
	X_train, y_train = shuffle(X_train,y_train)
	X_test, y_test, labels_test = shuffle(X_test,y_test,labels_test)
	labels_test = np.expand_dims(labels_test,axis=1)

	return X_train, y_train, X_test, y_test, labels_test, class_lookup


# get data from different train and test source
def get_training_data(train_path,sel_WL,use_t_int,data_augm_train,preprocessing_method,class_lookup):
	# load training from file
	X_train,y_train,labels_train,class_lookup = getData_binary(train_path,
															sel_WL=sel_WL,
															use_t_int=use_t_int,
															data_augm=data_augm_train,
															method=preprocessing_method,
															class_lookup=class_lookup)
	# shuffle to randomize
	X_train, y_train = shuffle(X_train,y_train)

	return X_train, y_train, class_lookup


def get_test_data(test_path,sel_WL,use_t_int,data_augm_test,preprocessing_method,class_lookup):
	# load test data from file
	X_test,y_test,labels_test,class_lookup = getData_binary(test_path,
															sel_WL=sel_WL,
															use_t_int=use_t_int,
															data_augm=data_augm_test,
															method=preprocessing_method,
															class_lookup=class_lookup)
	# shuffle to randomize
	X_test, y_test, labels_test = shuffle(X_test,y_test,labels_test)
	labels_test = np.expand_dims(labels_test,axis=1)

	return X_test, y_test, labels_test


# read in the data from files
def getData_binary(d_path,sel_WL=True,use_t_int=False,data_augm=False,method=3,class_lookup=class_lookup):
	X = []
	y = []
	labels = []
	s25 = []
	s60 = []
	count = []

	for filename in os.listdir(d_path):
		# get class from filename
		re_str = re.compile("\S*_([a-zA-Z0-9]*).json")
		class_name = re.findall(re_str,filename)
		try:
			cur_class = class_lookup[class_name[0]]
		except:
			warnings.warn('No valid class. Ignoring file %s' %(filename), Warning)
			input('Press [ENTER] to continue.')
			continue
		# open each json-file once
		with open(os.path.join(d_path,filename)) as f_json:
			data = json.load(f_json)
			# iterate over all campaigns
			for key in data.keys():
				# extract the spectralon measurements for the current class and campaign
				# further calculate the factor per wavelength so that the spectralon measurements correspond to their reflectance
				for ref in data[key]['reference'].keys():
					# only take wavelengths between range_low and range_high
					measurements = np.asarray(data[key]['reference'][ref][1])
					if sel_WL:
						wavelengths = np.asarray(data[key]['reference'][ref][0])
						ind = np.where( (wavelengths >= range_low) & (wavelengths <= range_high) )
						measurements = measurements[ind]
					if use_t_int:
						t_int = data[key]['reference'][ref][2]
						measurements = measurements / t_int

					if data[key]['reference'][ref][3] == 'spec_25':
						s25.append(measurements)
					elif data[key]['reference'][ref][3] == 'spec_60':
						s60.append(measurements)

				ref_scale, ref_curve = CalcRefScale(s25,s60)
				# iterate over all the epochs of the current campaign (without the reference measurements)
				cnt=0
				for ep in data[key].keys():
					if (ep != 'reference'):
						measurements = np.asarray(data[key][ep][1])
						# only take wavelengths between range_low and range_high
						if sel_WL:
							wavelengths = np.asarray(data[key][ep][0])
							ind = np.where( (wavelengths >= range_low) & (wavelengths <= range_high) )
							measurements = measurements[ind]
						if use_t_int:
							t_int = np.asarray(data[key][ep][2])
							measurements = measurements / t_int

						cur_x = pre_process(measurements,ref_curve,ref_scale,method)

						X.append(cur_x)
						y.append(cur_class)
						labels.append(filename[:-5])
						cnt += 1

						if data_augm:
							sigma = 0.005103 # determined empirically
							augm_noise = sigma * np.random.uniform(-1,1,(measurements.size,1))
							augm_measurements = np.expand_dims(measurements,axis=1) + np.reshape(augm_noise,(2998,1))
							augm_measurements = np.squeeze(augm_measurements,axis=1)

							augm_cur_x = pre_process(augm_measurements,ref_curve,ref_scale,method)

							X.append(augm_cur_x)
							y.append(cur_class)
							labels.append(filename[:-5]+"_augm")
							cnt += 1

				count.append(cnt)

	return np.array(X), np.array(y), labels, class_lookup



#%% Machine Learning Baselines
# Logistic Regression
def log_reg(X_train,y_train,X_test,y_test):
	logreg = LogisticRegression()
	logreg.fit(X_train, y_train)
	pred = logreg.predict(X_test)
	logreg_score = logreg.score(X_test, y_test)
	print('logreg: '+str(logreg_score))

	return logreg_score


# Linear Discriminant Analysis
def lda(X_train,y_train,X_test,y_test):
	lda = LinearDiscriminantAnalysis()
	lda.fit(X_train, y_train)
	lda_score = lda.score(X_test, y_test)
	print('LDA: '+str(lda_score))
	
	return lda_score


# Support Vector Machine
def svm(X_train,y_train,X_test,y_test):
	svm = SVC(kernel='linear')
	svm.fit(X_train, y_train)
	svm_score = svm.score(X_test, y_test)
	print('SVM: '+str(svm_score))
	
	return svm_score


# k Nearest Neighbor
def knn(X_train,y_train,X_test,y_test):
	knn = KNeighborsClassifier()
	knn.fit(X_train, y_train)
	knn_score = knn.score(X_test, y_test)
	print('kNN: '+str(knn_score))

	return knn_score




#%% Basic Neural Network
def basic_nn(X_train,y_train,X_test,y_test,class_lookup,dropout_rate,val_split,log1,labels_test):
	# adjust labels and determine shape 
	y_train = to_categorical(y_train, len(class_lookup))
	y_test = to_categorical(y_test, len(class_lookup))
	input_shape = X_train.shape[1:]
	# build model 
	inputs = Input(shape=input_shape)
	
	x = Dense(256, activation='relu')(inputs)
	x = Dropout(rate=dropout_rate)(x)
	x = Dense(128, activation='relu')(x)
	x = Dense(64, activation='relu')(x)
	#x = Dense(32, activation='relu')(x)

	# for binary models
	if len(class_lookup) == 2:
		outputs = Dense(len(class_lookup), activation='sigmoid')(x) 

		model = Model(inputs, outputs)
	
		# compile & fit model
		model.compile(loss='binary_crossentropy',
					  optimizer=Adam(learning_rate=lr_schedule(0)),
					  metrics=['accuracy'])

	# for more than 2 classes
	else:
		outputs = Dense(len(class_lookup), activation='softmax')(x) 

		model = Model(inputs, outputs)
	
		# compile & fit model
		model.compile(loss='categorical_crossentropy',
					  optimizer=Adam(learning_rate=lr_schedule(0)),
					  metrics=['accuracy'])

	model.summary()

	# NN callbacks
	if use_callbacks:
		es = EarlyStopping(monitor='val_loss',
						   patience=75,
						   restore_best_weights=True)
		mc = ModelCheckpoint(results_dir + '/models/' + cur_datetime + '_BasicNN.h5',
							 monitor='val_loss',
							save_best_only=True)
		lr_scheduler = LearningRateScheduler(lr_schedule)
		cb_list = [es,mc,lr_scheduler] 
		log1.write("Callbacks: EarlyStopping, ModelCheckpoint, LearningRateScheduler \n\n")

		# create .ini file with parameters
		params = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
		params['Preprocessing'] = {'sel_WL': sel_WL,
								   'range_low': range_low,
								   'range_high': range_high,
								   'method': preprocessing_method}
		with open(results_dir + '/models/' + cur_datetime + '_BasicNN.ini', 'w') as configfile:
			params.write(configfile)
		log1.write("ini-file created. \n\n")

	# log model architecture and parameters
	log1.write("Basic Neural Net \n #epochs: %i \n Batch size: %i \n Initial learning rate: %.5f \n Dropout rate: %.1f \n\n" % (epochs, batch_size, lr_schedule(0), dropout_rate))
	model.summary(print_fn=lambda x: log1.write(x + '\n'))
	log1.write("\n")

	history = model.fit(x=X_train,
						y=y_train,
						batch_size=batch_size,
						epochs=epochs,
						verbose=1, 
						callbacks=cb_list,
						validation_split=val_split,
						shuffle=True)

	# Plot training & validation accuracy values
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	#plt.show()
	plt.savefig(results_dir + '/output/'+cur_datetime+'_basicNN_accuracy.png')
	plt.close()

	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	#plt.show()
	plt.savefig(results_dir+'/output/'+cur_datetime+'_basicNN_loss.png')
	plt.close()

	# calculate loss and accuracy on test data set
	pred = model.predict(x=X_test)
	pred = np.concatenate((pred,labels_test),axis=1)
	score = model.evaluate(x=X_test, y=y_test)
	print('loss: {}, accuracy: {}'.format(score[0],score[1]))

	# log the results
	log1.write("#epochs trained for: %i \n" % (len(history.history['loss'])))
	log1.write("Loss: %.2f \nAccuracy: %.3f \n\n" % (score[0],score[1]))

	return score




#%% ResNet - Fully Connected 
def resnet_fc(X_train,y_train,X_test,y_test,batch_size,epochs,val_split,num_classes,num_filters,log1):
	# adjust labels and determine shape 
	y_train = to_categorical(y_train, num_classes)
	y_test = to_categorical(y_test, num_classes)
	input_shape = X_train.shape[1:]
	# build and compile
	model = build_resnet_fc(input_shape,num_classes=num_classes,num_filters=num_filters)

	# for binary models
	if num_classes == 2:
		model.compile(loss='binary_crossentropy',
					  optimizer=Adam(learning_rate=lr_schedule(0)),
					  metrics=['accuracy'])
	# for more than 2 classes
	else:
		model.compile(loss='categorical_crossentropy',
					  optimizer=Adam(learning_rate=lr_schedule(0)),
					  metrics=['accuracy'])

	model.summary()

	# NN callbacks
	if use_callbacks:
		es = EarlyStopping(monitor='val_loss',
						   patience=75,
						   restore_best_weights=True)
		mc = ModelCheckpoint(results_dir+'/models/' + cur_datetime + '_ResNet.h5',
							 monitor='val_loss',
							save_best_only=True)
		lr_scheduler = LearningRateScheduler(lr_schedule)
		cb_list = [es,mc,lr_scheduler]
		log1.write("Callbacks: EarlyStopping, ModelCheckpoint, LearningRateScheduler \n\n")

		# create .ini file with parameters
		params = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
		params['Preprocessing'] = {'sel_WL': sel_WL,
								   'range_low': range_low,
								   'range_high': range_high,
								   'method': preprocessing_method}
		with open(results_dir+'/models/' + cur_datetime + '_ResNet.ini', 'w') as configfile:
			params.write(configfile)
		log1.write("ini-file created. \n\n")
	
	# log model architecture and parameters
	log1.write("ResNet - fully connected \n #epochs: %i \n Batch size: %i \n Initial learning rate: %.5f \n Dropout rate: %i \n #filters: %i \n\n" % (epochs,batch_size,lr_schedule(0),dropout_rate,num_filters))
	model.summary(print_fn=lambda x: log1.write(x + '\n'))
	log1.write("\n")

	# fit model
	history = model.fit(x=X_train,
					 y=y_train,
					 batch_size=batch_size,
					 epochs=epochs,
					 verbose=1, 
					 callbacks = cb_list,
					 validation_split=val_split,
					 shuffle=True)

	# Plot training & validation accuracy values
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	#plt.show()
	plt.savefig(results_dir+'/output/'+cur_datetime+'_ResNetFC_accuracy.png')
	plt.close()

	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	#plt.show()
	plt.savefig(results_dir+'/output/'+cur_datetime+'_ResNetFC_loss.png')
	plt.close()

	score = model.evaluate(x=X_test, y=y_test)
	print('loss: {}, accuracy: {}'.format(score[0],score[1]))

	# log the results
	log1.write("#epochs trained for: %i \n" % (len(history.history['loss'])))
	log1.write("Loss: %.2f \nAccuracy: %.3f \n\n" % (score[0],score[1]))

	return score


# fully connected ResNet model
def build_resnet_fc(input_shape,num_classes,num_filters=num_filters):
	# model definition
	num_res_blocks = 1 # to fix depending on depth (num_res_blocks * 6 + 2)

	inputs = Input(shape=input_shape)
	x = resnet_layer(inputs=inputs)
	# instantiate the stack of residual units
	for stack in range(3):
		for res_block in range(num_res_blocks):
			strides = 1
			if stack > 0 and res_block == 0:  # first layer but not first stack
				# downsample
				strides = 2  
			y = resnet_layer(inputs=x,
					num_filters=num_filters)
			y = resnet_layer(inputs=y,
					num_filters=num_filters,
					activation=None)
			if stack > 0 and res_block == 0:  # first layer but not first stack
				# linear projection residual shortcut connection to match changed dims
				x = resnet_layer(inputs=x,
					 num_filters=num_filters,
					 activation=None,
					 batch_normalization=False)
			x = keras.layers.add([x, y])
			x = Activation('relu')(x)
		num_filters *= 2

	# add classifier on top
	# for binary models
	if num_classes == 2:
		outputs = Dense(num_classes,
					activation='sigmoid',
					kernel_initializer='he_normal')(x)
	# for more than 2 classes
	else:
		outputs = Dense(num_classes,
					activation='softmax',
					kernel_initializer='he_normal')(x)

	# instantiate model
	model = Model(inputs=inputs, outputs=outputs)
	
	return model


# definition of the residual layer
def resnet_layer(inputs,num_filters=num_filters,dropout_rate=dropout_rate,activation='relu',batch_normalization=True,conv_first=True):
	dense = Dense(num_filters,
			   kernel_initializer='he_normal',
			   kernel_regularizer=l2(1e-4))

	x = dense(inputs)
	if batch_normalization:
		x = BatchNormalization()(x)
	if activation is not None:
		x = Activation(activation)(x)
		x = Dropout(rate=dropout_rate)(x)

	return x


# systematically reduce learning rate
def lr_schedule(epoch):
	lr = 1e-3
	if epoch > 180:
		lr *= 0.5e-3
	elif epoch > 160:
		lr *= 1e-3
	elif epoch > 120:
		lr *= 1e-2
	elif epoch > 80:
		lr *= 1e-1
	print('Learning rate: ', lr)

	return lr




#%% call functions
# get data
if separate_data_sets:
	X_train,y_train,class_lookup = get_training_data(train_path,sel_WL,use_t_int,data_augm_train,preprocessing_method,class_lookup)
	X_test,y_test,labels_test = get_test_data(test_path,sel_WL,use_t_int,data_augm_test,preprocessing_method,class_lookup)
	log1.write("Data from separate training and test files. \n %i training samples, %i test samples. \n" % (np.shape(y_train)[0], np.shape(y_test)[0]))
	log1.write("Preprocessing method: %i \n" % (preprocessing_method))
else:
	X_train,y_train,X_test,y_test,labels_test,class_lookup = get_data(train_path,sel_WL,use_t_int,data_augm_train,train_size,preprocessing_method,class_lookup)
	log1.write("Data from one dataset with train-test split. \n %i training samples, %i test samples. \n" % (np.shape(y_train)[0], np.shape(y_test)[0]))
	log1.write("Preprocessing method: %i \n" % (preprocessing_method))
if preprocessing_method==1:
	# standardization
	mean_X = np.mean(X_train)
	std_X = np.std(X_train)
	X_train = (X_train - mean_X) / std_X
	X_test = (X_test - mean_X) / std_X

# get Machine Learning Baselines
if ML_baselines:
	logreg_score = log_reg(X_train,y_train,X_test,y_test)
	lda_score = lda(X_train,y_train,X_test,y_test)
	svm_score = svm(X_train,y_train,X_test,y_test)
	knn_score = knn(X_train,y_train,X_test,y_test)
	log1.write("Machine Learning Baselines \n Logistic Regression: %.3f \n Linear Discriminant Analysis: %.3f \n Support Vector Machine: %.3f \n k Nearest Neighbors: %.3f \n\n" % (logreg_score, lda_score, svm_score, knn_score))

# get Basic NN
if basic_model:
	basic_nn_score = basic_nn(X_train,y_train,X_test,y_test,class_lookup,dropout_rate,val_split,log1,labels_test)

# get ResNet fully connected
if resnet_model:
	resnet_fc_score = resnet_fc(X_train,y_train,X_test,y_test,batch_size,epochs,val_split,len(class_lookup),num_filters,log1)



