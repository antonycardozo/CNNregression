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
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate


def create_cnn(width, height, depth, filters=(32, 64, 128, 256), regress=False):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1

    inputs = Input(shape=inputShape)
    x = inputs

    # Build the convolutional base
    for f in filters:
        x = Conv2D(f, (3, 3), padding="valid")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
		# CONV => RELU => BN => POOL

                
    # CONV => RELU => BN => PO
    # flatten the volume, then FC => RELU => BN => DROPOUT
    x1 = GlobalAveragePooling2D()(x)
    x2 = GlobalMaxPooling2D()(x)
    x = Concatenate()([x1,x2])
    x = Dense(128)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
	# apply another FC layer, this one to match the number of nodes
	# coming out of the MLP

    x = Dense(64)(x)
    x = Activation("relu")(x)
    
    x = Dense(32)(x)
    x = Activation("relu")(x)
    if regress:
        x = Dense(1, activation="softplus")(x)
	# construct the CNN
    model = Model(inputs, x)
	# return the CNN
    return model    


