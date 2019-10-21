
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Activation 



#%% VGG16
model_vgg16 = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling='max', classes=10)
model_vgg16.summary()

model = 


##%% ResNet50
#model_resnet50 = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling='max', classes=10)
#model_resnet50.summary()