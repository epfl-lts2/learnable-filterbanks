# Helena Peic Tukuljac and Benjamin Ricaud, 2019
# In this file we define our new layer.
# The layer takes sound in time domain as input.
# The output is processed sound by the learned filter bank.

from tensorflow.keras import activations, initializers, regularizers, constraints
import tensorflow as tf
import learnable_filterbanks.kernels as kernels
import numpy as np

logger = tf.get_logger()

class Pos(constraints.Constraint):
    #Constrain the weights to be strictly positive (add epsilon), so they don't collapse to zero
    def __call__(self, p):
        p1 = tf.math.abs(p)
        epsilon = tf.keras.backend.epsilon()
        p2 = p1 + tf.cast(p1 < epsilon, tf.keras.backend.floatx())*epsilon
        return p2

class Pos_and_f_ordered(constraints.Constraint):
    #Constrain the weights to be strictly positive (add epsilon), so they don't collapse to zero
    def __call__(self, p):
        # positivity
        p1 = tf.math.abs(p)
        # sorting along the first parameter (assuming this parameter is the frequency)
        epsilon = tf.keras.backend.epsilon()
        p2 = p1 + tf.cast(p1 < epsilon, tf.keras.backend.floatx())*epsilon
        p3 = tf.gather(p2,tf.argsort(p2[:,0]),axis=0)
        return p2


class FilterConvolution(tf.keras.layers.Layer):
    """Implements a learnable filterbank layer, using parameterized filters.
    """
  
    
    def __init__(self, filter_type, filter_number=32, init='uniform', activation='linear',
                 padding='same', strides=1, data_format='channels_last', use_bias=False,
                 bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=Pos(), bias_constraint=None,
                 input_shape=None, **kwargs):

        # TODO: support TF 2.x conventions as well
        if padding.lower() not in ['valid', 'same']:
            raise ValueError('Invalid border mode for FilterConvolution:', padding)
        if data_format.lower() not in ['channels_first', 'channels_last']:
            raise ValueError('Invalid data format for FilterConvolution:', data_format)

        logger.info('Creating FilterConvolution layer with %s', filter_type)
        self.filter_number = filter_number
        self.filter_type = filter_type
        

        ### Initializer
        class KernelInitializer(initializers.Initializer):
            """Initializer that generates tensors initialized from kernel function.
            """

            def __call__(self, shape, dtype=None):
                logger.info('initializing kernel with shape %s',shape)
                return filter_type.init_filter_bank(shape)

        self.kernel_initializer = KernelInitializer()

        self.activation = activations.get(activation)
        self.padding = padding
        self.tf_padding = padding.upper()
        self.strides = strides
        self.subsample = (1,strides)#, 1)
        self.data_format = data_format.lower()
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.use_bias = use_bias        
        super(FilterConvolution, self).__init__(**kwargs)

    def build(self, input_shape):
        # get dimension and length of input
        if self.data_format == 'channels_first':
            self.input_dim = input_shape[1]
            self.input_length = input_shape[2]
            self.tf_format = 'NCHW'
        else:
            self.input_dim = input_shape[2]
            self.input_length = input_shape[1]
            self.tf_format = 'NHWC'
        # initialize and define filter widths
        self.parameter_number = self.filter_type.PARAMETER_NUMBER
        ### Weight definition
        kernel_shape = (self.filter_number,self.parameter_number)
        self.filters = self.filter_number
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        super(FilterConvolution, self).build(input_shape)
        
        
    def call(self, x):
        """[summary]
        
        Args:
            x (Tensor): if data_format is 'channels_first' then
                3D tensor with shape: `(batch_samples, input_dim, steps)`.
                        if data_format is 'channels_last' then
                3D tensor with shape: `(batch_samples, steps, input_dim)`.

        Raises:
            ValueError: if the input audio is not mono, i.e. input_dim != 1

        Returns:
            [Tensor]: if data_format is 'channels_first' then
                4D tensor with shape: `(batch_samples, output_dim, new_steps, filter_number)`.
                `steps` value might have changed due to padding.
                     if data_format is 'channels_last' then
                4D tensor with shape: `(batch_samples, new_steps, filter_number, output_dim)`.
                `steps` value might have changed due to padding.
                     output_dim is squeezed out in case of real filters
                     
        """
        if self.input_dim > 1:
            raise ValueError('This layer accept only mono audio signals.')

        # shape of x is (batches, input_dim, input_len) if 'channels_first'
        # shape of x is (batches, input_len, input_dim) if 'channels_last'
        # we reshape x to channels first for computation
        if self.data_format == 'channels_last':
            x = tf.transpose(x, (0, 2, 1))

        kernels = [self.filter_type.get_kernel(self.kernel[i]) for i in range(self.filter_number)]
        kernels = tf.stack(kernels, axis=0)
        
        if not self.filter_type.COSINE_AND_SINE_FILTER:
            kernels = tf.expand_dims(kernels, 0)
            kernels = tf.transpose(kernels,(0, 2, 3, 1))
        else:
            kernels = tf.transpose(kernels,(2, 1, 3, 0)) # shape: (1, filter_length, 1, num_filters)

        # reshape input s.t. number of dimensions is first (before batch dim)
        x = tf.transpose(x, (1, 0, 2))

        def gen_conv(x_slice):
            x_slice = tf.expand_dims(x_slice, 1) # shape (num_batches, 1, input_length)
            x_slice = tf.expand_dims(x_slice, 2) # shape (num_batches, 1, 1, input_length)
            return tf.nn.conv2d(x_slice, kernels, strides=self.subsample, padding=self.tf_padding, data_format='NCHW')

        
        outputs = gen_conv(x[0,:,:])        

        if self.filter_type.COSINE_AND_SINE_FILTER:
            # Replace the channel by 2 channels of cos and sin
            outputs = tf.squeeze(outputs,axis=2)
            outputs = tf.reshape(outputs, shape=[-1, self.filter_number//2, 2, outputs.shape[2]])
        
        #------
        #output must be (batch,channels,filters_numbers,new_steps)
        #------
        outputs = tf.transpose(outputs, (0, 2, 3, 1))
        if self.data_format == 'channels_last':
            outputs = tf.transpose(outputs,(0, 2, 3, 1))

        if self.use_bias:
            outputs = tf.nn.bias_add(
                outputs,
                self.bias,
                data_format=self.tf_format)
        if self.activation is not None:
            outputs =  self.activation(outputs)
        
        if not self.filter_type.COSINE_AND_SINE_FILTER:
            sq_dim = 1 if self.data_format == 'channels_first' else 3
            return tf.squeeze(outputs, sq_dim)
        return outputs 

    
    def get_config(self):
        config = {'filter_number': self.filter_number,
                  #'init': self.kernel_initializer.__name__,
                  'activation': self.activation.__name__,
                  'padding': self.padding,
                  'strides': self.strides,
                  'data_format': self.data_format,
                  'kernel_regularizer': self.kernel_regularizer.get_config() if self.kernel_regularizer else None,
                  'bias_regularizer': self.bias_regularizer.get_config() if self.bias_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'kernel_constraint': self.kernel_constraint.get_config() if self.kernel_constraint else None,
                  'bias_constraint': self.bias_constraint.get_config() if self.bias_constraint else None,
                  'use_bias': self.use_bias}
        base_config = super(FilterConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




################################################
class Modulus(tf.keras.layers.Layer):
    """ This layer compute the modulus of the input values,
        assuming the filters are ordered by alterning real and imaginary parts filters:
        modulus = sqrt( real_part**2 + imag_part**2 )

    # Input shape
        if data_format is 'channels_first' then
            4D tensor with shape: `(batch_samples, input_dim, steps, filter_number)`.
        if data_format is 'channels_last' then
            4D tensor with shape: `(batch_samples, steps, filter_number, input_dim)`.        
    # Output shape
        3D tensor with shape: `(batch_samples, steps, filter_number/2)`.
        
    # Options
        logscale: if True, output log( 1 + modulus ). Default False.
        data_format: specify the location of the channels in the tensor. Can be 'channel_last' or
        'channel_first'. Default 'channel_last'.
    """
    
    def __init__(self, logscale=False, data_format='channels_last', **kwargs):
        self.logscale = logscale
        self.data_format= data_format.lower()
        super(Modulus, self).__init__(**kwargs)

    def call(self, inputs):
        if self.data_format == 'channels_last':
            reals = inputs[:, :, :, 0]
            imags = inputs[:, :, :, 1]
            reals = tf.expand_dims(reals, 3)
            imags = tf.expand_dims(imags, 3)
        else:
            reals = inputs[:, 0, :, :]
            imags = inputs[:, 1, :, :]
            reals = tf.expand_dims(reals, 1)
            imags = tf.expand_dims(imags, 1)

        modulus =  tf.sqrt(tf.square(reals) + tf.square(imags) + tf.keras.backend.epsilon())
        if self.data_format == 'channels_first':
            output = tf.squeeze(modulus, 1)
        else:
            output = tf.squeeze(modulus, 3)
        if self.logscale:
            return tf.math.log(1 + output)
        else:
            return output

    def get_config(self):
        config = {'logscale': self.logscale,
                  'data_format': self.data_format}
        base_config = super(Modulus, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))