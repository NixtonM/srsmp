import numpy as np
import os
import configparser
from .Preprocessings import pre_process
from keras.models import load_model
from keras.optimizers import Adam


class Hydrophobic_Predictor:
	def __init__(self,config_file='config.ini'):
		# initialize config and params parser
		self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
		self.params = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())

		# load config.ini
		self.config.read('config.ini')
		model_dir = self.config['PredictApp']['model_dir']
		files = os.listdir(model_dir)
		# check if one .ini and one .h5 file present
		if len(files) != 2:
			raise RuntimeError('Make sure only two files are given. (The weights as .h5 and the corresponding .ini file.)')

		# load model and corresponding .ini file
		try:
			if files[0].endswith(".h5"):
				self.model = load_model(model_dir+'/'+files[0])
				self.params.read(model_dir+'/'+files[1])
			elif files[0].endswith(".ini"):
				self.params.read(model_dir+'/'+files[0])
				self.model = load_model(model_dir+'/'+files[1])
		except:
			raise RuntimeError('Make sure the weights are given as .h5 and with the corresponding .ini file.')

		# compile the model
		self.model.compile(loss='binary_crossentropy',
					  optimizer=Adam(learning_rate=1e-5),
					  metrics=['accuracy'])


	def set_spectralon(self,spec,wavelengths,spec_number):
		self.ref_curve = self.cut_to_range(spec,wavelengths)
		self.ref_curve = self.ref_curve / max(abs(self.ref_curve))
		self.ref_scale = (spec_number/100) / self.ref_curve


	def cut_to_range(self,measurements,wavelengths):
		if self.params['Preprocessing']['sel_WL']:
			ind = np.where( (wavelengths >= int(self.params['Preprocessing']['range_low'])) 
                  & (wavelengths <= int(self.params['Preprocessing']['range_high'])) )
			measurements = measurements[ind]
		
		return measurements


	def preprocess(self,x):
		cur_x = pre_process(x,self.ref_curve,self.ref_scale,self.params['Preprocessing']['method'])

		return cur_x


	def predict(self,measurements,wavelengths):
		measurements_cut = self.cut_to_range(measurements,wavelengths)
		measurements_preprocessed = self.preprocess(measurements_cut)
		# {'treated': 0, 'untreated': 1}
		measurements_preprocessed = np.expand_dims(measurements_preprocessed,axis=0)
		pred = self.model.predict(measurements_preprocessed)
		prediction = np.argmax(pred)

		return prediction
