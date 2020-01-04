Train the Model
===============
Different options can be chosen previous to training the model. These include options about the 
training data used, the pre-processing method applied and the model used for training. The 
different options are introduced in the following parts.

Pre-processing
--------------
Different operations on the measurements can be enabled.

:code:`sel_WL = True`
   Cuts the wavelength range to the range of range_low - range_high as specified in the 
   spectrometers specification.
:code:`range_low` & :code:`range_high`
   Specify the wavelength range to be used in nanometers.
:code:`use_t_int = True`
   Uses the integration time to scale the measurement accordingly, re-introducing the average 
   intensity.
:code:`data_augm_train = True`
   Each measurement is duplicated by applying noise to each wavelength. The noise is scaled 
   between -1s and 1s. The s is determined empirically.
:code:`data_augm_test = False`
   Normally set to False as test data is not augmented, normally.
:code:`preprocessing_method`
   Different methods (numbered 1 - 4) are available to preprocess the measurements.

   #. The measurement is multiplied by the reference scale and then scaled to 1 and 
      standardize.
   #. The measurement is scaled to 1 and the reference curve (as well scaled to 1) is 
      subtracted. The property assessed is the difference.
   #. The measurement is scaled to 1 and the reference curve (as well scaled to 1) is 
      subtracted. A MinMaxScaler is applied on the differences.
   #. The measurement is scaled to 1 and the reference curve (as well scaled to 1) is 
      subtracted. A RobustScaler is applied on the differences.

   The reference scale and curve are calculated from the given reference measurements. Knowing 
   the reflectivity of the reference, a factor for each wavelength can be calculated to 
   translate the measured intensity to the expected intensity. The reference scale is then an 
   array of all those factors. The reference curve is built up by calculating the mean value per 
   wavelength over all reference measurements of the current epoch and scaling it to 1.

NN parameter selection
----------------------
Different parameters have to be defined to run the neural networks.

:code:`train_size`
   Scalar defining the percentage of the data used for training. (Only applicable if one data 
   set is used.)
:code:`batch_size`
   |space|
:code:`epochs`
   |space|
:code:`val_split`
   Scalar defining the percentage of training data used for validation.
:code:`dropout_rate`
   |space|
:code:`use_callbacks`
   Whether or not to use callbacks (EarlyStopping, ModelCheckpoint, LearningRateScheduler).
:code:`num_filters`
   Scalar defining the number of filters used in the fully connected layers for the ResNet8 
   architecture.

Training data
-------------
The files to train the model are built up in a dictionary structure and are saved as .json files. 
They consist of measurement campaigns on the first level. Each campaign consists of a number of 
epochs and a reference. Each epoch holds an array with the wavelengths, an array with the measured 
intensities and a scalar giving the integration time used. The reference itself can again hold 
multiple reference epochs holding the wavelengths, the measured intensities, the integration time 
used and a string specifying the reference used.

For the training data, two options are available.

#. All data is in one file (separate_data_sets = False)
#. Training and test data are in different files (separate_data_sets = True)

For the first option, the parameter separate_data_sets is set to True and the data path to the 
file is given as the varable data_path. The data is then read from the specified file and split 
into a training and test data set using the specified train ratio train_size. 

The second option reads in the training and test data separately and no split of the data set is 
carried out.

In both options, each measurement is augmented if the option data_augm or data_augm_train is set 
to True. Further, the wavelength range is cut to range_low - range_high and the data is 
pre-processed directly.

Model selection
---------------
Different models can be trained. 

#. Simple machine learning models as a baseline (ML_baselines)
#. A basic neural network (basic_model)
#. An adjusted ResNet architecture (resnet_model)

The machine learning baselines include Logistic Regression, Linear Discriminant Analysis, Support 
Vector Machine and k-Nearest Neighbors. They suit as a reference and give a hint on how well the 
Neural Networks are expected to perform. No additional parameters have to be set for these models. 
They directly take the preprocessed measurements as input.

The basic Neural Network consists of a fully connected (or dense) layer with 256 units, a dropout 
layer with a dropout rate specified in dropout_rate and two fully connected layers with 128 and 
64 units, respectively. 

In a third approach, a ResNet8 is adjusted by exchanging its convolutional layers with dense layers. 

Both neural networks use the ReLU (rectified linear unit) activation function for all fully 
connected layers. The last layer is a fully connected layer with the number of units corresponding 
to the number of classes present. It uses the Sigmoid activation function as the present case is 
a binary classification. If the approach is changed to classify more than two classes, the 
activation has to be changed to Softmax activation function. The model uses binary-crossentropy 
loss and the Adam optimizer with a learning rate schedule which adjusts the learning rate depending 
on the current epoch, starting with a learning rate of 0.003. If the variable use_callbacks is set to 
True, the model uses the following callbacks.

- Early stopping: Monitors the validation loss and stops training if the loss has not improved over the last 75 epochs.
- Model checkpoint: Evaluates the validation loss and saves the model of the current epoch if it has the lowest validation loss.
- Learning rate scheduler: Adjusts the learning rate depending on the current epoch starting with a learning rate of 0.003.

If callbacks are used and hence the best model is saved, an .ini file is saved with it for successful 
parameter initialization and usage when predicting.
After training, two plots with the training and test accuracy and loss, respectively, are saved for visual evaluation.


.. |space| unicode:: U+0020