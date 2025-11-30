# import the necessary packages
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import GlobalAveragePooling2D


def create_cnn(width, height, depth, filters=(32, 64, 128, 512), regress=False):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1
    
    # define the model input
	# define the model input
    inputs = Input(shape=inputShape)
	# loop over the number of filters
    for (i, f) in enumerate(filters):
		# if this is the first CONV layer then set the input
		# appropriately
        if i == 0:
            x = inputs
            x = Conv2D(f, (3, 3), padding="valid")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
        else:
            x = Conv2D(f, (3, 3), padding="valid")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
		# CONV => RELU => BN => POOL

                
    # CONV => RELU => BN => PO
    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = GlobalAveragePooling2D()(x)
    x = Dense(512)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
	# apply another FC layer, this one to match the number of nodes
	# coming out of the MLP
 
    x = Dense(128)(x)
    x = Activation("relu")(x)
	# check to see if the regression node should be added
    x = Dense(64)(x)
    x = Activation("relu")(x)
    if regress:
        x = Dense(1, activation="sigmoid")(x)
	# construct the CNN
    model = Model(inputs, x)
	# return the CNN
    return model    


