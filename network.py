import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout
from tensorflow.keras.regularizers import l2

def model(loss='mse', optimizer='adam'):
    model = Sequential([
    
        # Normalization Layer
        Lambda(lambda x:  (x / 127.5) - 1., input_shape=(70, 160, 3)),

        # Conv Layer 1
        Conv2D(filters = 24, 
               kernel_size = 5, 
               strides = 2, 
               activation="elu",
               kernel_regularizer=l2(0.001)),

        # Conv Layer 2
        Conv2D(filters = 36, 
               kernel_size = 5, 
               strides= 2, 
               activation="elu",
               kernel_regularizer=l2(0.001)),

        # Conv Layer 3
        Conv2D(filters = 48, 
               kernel_size = 5, 
               strides= 2, 
               activation="elu",
               kernel_regularizer=l2(0.001)),

        # Conv Layer 4
        Conv2D(filters = 64, 
               kernel_size = 3, 
               strides= 1, 
               activation="elu",
               kernel_regularizer=l2(0.001)),

        # Conv Layer 5
        Conv2D(filters = 64, 
               kernel_size = 3, 
               strides= 1, 
               activation="elu",
               kernel_regularizer=l2(0.001)),

        # Flatten Layer
        Flatten(),

        # Fully-connected Layer 1
        Dense(units = 100, activation="elu", kernel_regularizer=l2(0.001)),
       #  Dropout(0.2),

        # Fully-connected Layer 2
        Dense(units = 50, activation="elu", kernel_regularizer=l2(0.001)),
       #  Dropout(0.2),
        
        # Fully-connected Layer 3
        Dense(units = 10, activation="elu", kernel_regularizer=l2(0.001)),
       #  Dropout(0.2),
        
        # Output Layer
        Dense(units = 1),
    ])
    
    # Configure learning process with an optimizer and loss function
    model.compile(loss=loss, optimizer=optimizer)

    return model