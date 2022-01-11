import numpy as np
import tensorflow as tf
from learnable_filterbanks import filterconvolution, kernels
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Dropout, MaxPooling1D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Flatten, Activation, BatchNormalization
import gin


@gin.configurable
def audiomnist_audionet(num_classes=10):
    """AudioNet baseline model.

    Args:
        num_classes (int, optional): [description]. Defaults to 10.

    Raises:
        ValueError: [description]

    Returns:
        Model: AudioNet model for AudioMNIST digit classification (cf. https://arxiv.org/abs/1807.03418)
    """
    raw_input = Input(shape=(8000,1,))
    x = Conv1D(100, 3, padding='same', activation='relu')(raw_input)
    x = MaxPooling1D(3, strides=2)(x)

    x = Conv1D(64, 3, padding='same', activation='relu')(x)
    x = MaxPooling1D(2, strides=2)(x)

    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = MaxPooling1D(2, strides=2)(x)

    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = MaxPooling1D(2, strides=2)(x)

    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = MaxPooling1D(2, strides=2)(x)

    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = MaxPooling1D(2, strides=2)(x)
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Dropout(0.5)(x)

    x = Dense(512)(x)
    x = Dropout(0.5)(x)
    if num_classes == 10:
        out = Dense(10, activation='softmax')(x)
        model = Model(inputs=[raw_input], outputs=[out])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    elif num_classes == 2: # for genre classification
        out = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=[raw_input], outputs=[out])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        raise ValueError("Invalid number of classes")
    return model

@gin.configurable
def audiomnist_fb_audionet(fs, input_len, overlap=75, num_filters=64, filter_type=5, num_classes=10):
    """AudioNet model with LF input layer for AudioMNIST digit classification

    Args:
        fs (int): Input sampling frequency (Hz)
        input_len (int): Length of input signal
        overlap (int, optional): Percentage of overlap for convolution in LF layer. Defaults to 75.
        num_filters (int, optional): Number of filters in LF layer. Defaults to 64.
        filter_type (int, optional): Filter id. Defaults to 5 (i.e. Gammatone fixed order 4).
        num_classes (int, optional): Number of classes. Defaults to 10.

    Returns:
        Model: AudioNet with LF input layer
    """
    filter_length = int(fs/100)
    fb_stride = int((100 - overlap)*filter_length/100)
    raw_input = Input(shape=(1,input_len))
   

    filters = kernels.create_filter_layer(filter_type, fs, filter_length)
    
    raw_input = Input(shape=(input_len, 1))
    x = filterconvolution.FilterConvolution(filter_number=num_filters, filter_type=filters,
                                            padding='same', data_format='channels_last', 
                                            activation='relu', strides = fb_stride)(raw_input)
    if filters.COSINE_AND_SINE_FILTER:
        x = filterconvolution.Modulus(data_format='channels_last', logscale=True)(x)
    
    x = Conv1D(64, 3, padding='same', activation='relu', data_format='channels_last')(x)
    x = MaxPooling1D(2, strides=2)(x)

    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = MaxPooling1D(2, strides=2)(x)

    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = MaxPooling1D(2, strides=2)(x)

    x = Flatten()(x)
    x = Dense(512)(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256)(x)
    x = Dropout(0.5)(x)

    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=[raw_input], outputs=[out])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=False)
    return model

# smaller network, non-audionet based
@gin.configurable
def audiomnist_fb_custom(fs, input_len, num_filters=32, overlap=75, filter_type=5):
    """SampleCNN-based model with LF layer for AudioMNIST classification

    Args:
        fs (int): Input sampling frequency (Hz)
        input_len (int): Length of input signal
        overlap (int, optional): Percentage of overlap for convolution in LF layer. Defaults to 75.
        num_filters (int, optional): Number of filters in LF layer. Defaults to 64.
        filter_type (int, optional): Filter id. Defaults to 5 (i.e. Gammatone fixed order 4).
        num_classes (int, optional): Number of classes. Defaults to 10.

    Returns:
        Model: SampleCNN with LF input layer
    """
    filter_length = int(fs/100) # 10ms
    fb_stride = int((100 - overlap)*filter_length/100)
    filters = kernels.create_filter_layer(filter_type, fs, filter_length)
    
    raw_input = Input(shape=(1,input_len))
    x = filterconvolution.FilterConvolution(filter_number=num_filters, filter_type=filters, padding='same',
                                            data_format='channels_first', activation='relu', 
                                            strides = fb_stride)(raw_input)
    if filters.COSINE_AND_SINE_FILTER:
        x = filterconvolution.Modulus(data_format='channels_first', logscale=True)(x)
    
    x = Conv1D(32, 32, strides=2, padding='same', activation='relu', data_format='channels_last')(x)
    x = MaxPooling1D(4, strides=4)(x)

    x = Conv1D(64, 16, strides=2, padding='same', activation='relu')(x)
    x = Conv1D(128, 8, strides=2, padding='same', activation='relu')(x)
    x = Conv1D(256, 4, strides=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(4, strides=4)(x)
    
    x = Flatten()(x)
    x = Dense(128)(x)
    x = Dropout(0.5)(x)

    x = Dense(64)(x)
    x = Dropout(0.5)(x)

    out = Dense(10, activation='softmax')(x)
    model = Model(inputs=[raw_input], outputs=[out])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



def conv_block(x,
               num_filters: int,
               activation = 'relu',
               normalization_layer=tf.keras.layers.BatchNormalization,
               dropout: float = 0.0,
               max_pooling: bool = True):
    
    x = Conv2D(filters=num_filters, kernel_size=[3, 1], activation=activation, padding='SAME')(x)
    if normalization_layer is not None:
        x = normalization_layer()(x)
    if dropout > 0.0:
        x = Dropout(dropout)(x)
    x = Conv2D(filters=num_filters, kernel_size=[1, 3], padding='SAME', activation=activation)(x)
   
    if normalization_layer is not None:
        x = normalization_layer()(x)
    if dropout > 0.0:
        x = Dropout(dropout)(x)
    if max_pooling:
        x = MaxPooling2D()(x)
        
    return x

@gin.configurable
def ConvNet(blk_filters, input_len, fs, overlap=75, num_filters=32, filter_length=None, num_classes=35, filter_type=5):
    """Create a ConvNet with a LF input layer (cf. https://github.com/google-research/leaf-audio)
    Used in the Google speech command experiment.


    Args:
        blk_filters (array): array of ints representing the size of each convolutional block
        input_len (int): Length of input signal.
        fs (int): Sampling frequency of input signal (Hz)
        overlap (int, optional): Percentage of overlap for convolution in LF layer. Defaults to 75.
        num_filters (int, optional): Number of filters in LF layer. Defaults to 32.
        filter_length (int, optional): Length of filters in LF layer. Defaults to 10ms equivalent if None.
        num_classes (int, optional): Number of output classes. Defaults to 35.
        filter_type (int, optional): Filter type id for LF layer. Defaults to 5 (Gammatone fixed order 4).

    Returns:
        [type]: [description]
    """
    if not filter_length:
        filter_length=int(fs/100) #10ms
    fb_stride = int((100 - overlap)*filter_length/100)
    num_features = num_filters
    filters = kernels.create_filter_layer(filter_type, fs, filter_length)
    
    # Variable-length input for feature visualization.
    x_in = Input(shape=(1, input_len), name='input')

    x = filterconvolution.FilterConvolution(filter_number=num_filters, filter_type=filters,
                                            padding='same', data_format='channels_first', 
                                            activation='relu', strides = fb_stride, name='sb')(x_in)
    
    x = BatchNormalization(name='norm0')(x)
    x = Activation('relu', name='relu0')(x)
    x = tf.expand_dims(x, axis=3)
    for (i, depth) in enumerate(blk_filters):
        x = conv_block(x, depth, max_pooling=(not i or i % 2))
    x = GlobalMaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(num_classes, kernel_initializer='glorot_uniform', name='logit')(x)
    x = Activation('softmax', name='pred')(x)
    model = Model(inputs=[x_in], outputs=[x], name='sample_cnn')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model    
    

