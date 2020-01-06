Train the Model
===============
Different options can be chosen previous to training the model. These include options about the 
training data used, the pre-processing method applied and the model used for training. The 
different options are introduced in the following parts.

This *How To* lets the user train a machine learning model to later be able to predict material 
properties.

Training Data
-------------
Before training the model, training data has to be acquired as described in *Acquire Training Data*.

For the training data, two options are available.

#. All data acquired are used for training and testing.
#. Specific files are used for training and testing only, respectively.

For option 1 (all data used), copy all files to the folder *data/training*. 
For option 2 (specific files), copy all files for training to the folder *data/training* and all files 
for testing to the folder *data/test*.

Adjust Variables
----------------
Different parameters have to be set to successfully train the model.

Open file: `repo/scripts/02_train_Predictor.py` in the editor to adjust the header variables.

What to train: 
   :code: `separate_data_sets: Boolean`
      *True* if specific files are used for training and test (option 2 above).
      *False* if all files are used for training and test (option 1 above).
   :code: `ML_baselines: Boolean`
      Set to *True* if the machine learning baselines should be trained, too. (Includes Support Vector
      Machine, Linear Discriminant Analysis, k-Nearest Neighbors and Logistic Regression.)
   :code: `basic_model: Boolean`
      Set to *True* if the basic artificial neural network with three fully connected layers should be
      trained.
   :code: `resnet_model: Boolean`
      Set to *True* if the adjusted ResNet8 should be trained.

Data pre-processing:
   :code:`sel_WL: Boolean`
      Set to *True* the wavelength range is cut to the range of *range_low - range_high* (E.g. the 
      range specified in the spectrometers specification.)
   :code:`range_low: Int` & :code:`range_high: Int` (To be set if code:`sel_WL=True`.)
      Specifies the wavelength range to be used in nanometers.
   :code:`use_t_int: Boolean`
      Set to *True* the integration time is used to scale the measurement accordingly, re-introducing 
      the average intensity.
   :code:`data_augm_train: Boolean`
      Set to *True* each measurement is duplicated by applying noise to each wavelength. The noise is 
      scaled between -1s and 1s. The s is determined empirically.
   :code:`data_augm_test: Boolean`
      Normally set to *False* as test data is not augmented. If set to *True* duplicates each measurement 
      in the same fashion as the training data.
   :code:`preprocessing_method: Int`
      Different methods (numbered 1 - 4) are available to preprocess the measurements.

      #. The measurement is multiplied by the reference scale and then scaled to 1 and 
         standardized.
      #. The measurement is scaled to 1 and the reference curve (as well scaled to 1) is 
         subtracted. The property assessed is the difference.
      #. The measurement is scaled to 1 and the reference curve (as well scaled to 1) is 
         subtracted. A MinMaxScaler is applied on the differences.
      #. The measurement is scaled to 1 and the reference curve (as well scaled to 1) is 
         subtracted. A RobustScaler is applied on the differences.

      The reference scale and curve are calculated from the given reference measurements. 

Neural network:
   :code:`train_size: Float`
      Scalar defining the percentage of the data used for training. (Only applicable if option 1
      is chosen.)
   :code:`batch_size: Int`
      |space|
   :code:`epochs: Int`
      Specifies the maximum number of epochs the model is trained for.
   :code:`val_split: Float`
      Scalar defining the percentage of training data used for validation.
   :code:`dropout_rate: Float`
      Scalar defining the percantage of neuronal connections erased.
   :code:`use_callbacks: Boolean`
      Whether or not to use callbacks (EarlyStopping, ModelCheckpoint, LearningRateScheduler,
	  ReduceLROnPlateau).
   :code:`num_filters: Int`
      Scalar defining the number of filters used in the fully connected layers for the ResNet8 
      architecture.

.. |space| unicode:: U+0020