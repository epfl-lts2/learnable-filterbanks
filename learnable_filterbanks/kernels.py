import numpy as np
from random import sample 
import scipy.special

import tensorflow as tf

class Filter:
    """Base class for filters composing a filterbank.
    """
    def __init__(self, fs, filter_type, cosine_and_sine_filter, parameter_number, parameter_names):
        """Filter initializer. 

        Args:
            fs: Sampling frequency in Hz
            filter_type: string representing the filter type
            cosine_and_sine_filter (bool): if true, the filter will return a complex output, real (using cosine modulation) otherwise 
            parameter_number (int): number of learnable parameters
            parameter_names (array): array containing the names of learnable parameters
        """
        self.fs = fs
        self.type = filter_type
        self.cosine_and_sine_filter = cosine_and_sine_filter        
        self.parameter_number = parameter_number
        self.parameter_names = parameter_names
    def get_kernel(self, parameters):
        pass
    def init_filter_bank(self, nb_filters):
        pass
    def visualize_filter_bank(self, figure, parameters, in_time = False, num_filters_in_time = 1):
        nb_filters, nb_parameters = parameters.shape
        for i in np.arange(nb_filters):
            sess = tf.compat.v1.Session()
            with sess.as_default() as s:
                k = self.get_kernel(parameters[i,:])
                if self.cosine_and_sine_filter:
                    kernel = tf.make_ndarray(k)[0]
                else:
                    kernel = tf.make_ndarray(k)
            if(in_time):
            # TO CHECK!
                if (num_filters_in_time*self.PARAMETER_NUMBER > i):
                    t = np.arange(1/self.fs, (len(kernel) + 1)/self.fs, 1/self.fs)
                    figure.plot(t, kernel)
                    figure.set_xlabel('time [s]')
            else:
                spectrum = np.abs(np.fft.rfft(kernel, axis=0))
                f = np.reshape(np.arange(0, (0.5 + 1/len(kernel)), 0.5/len(kernel)*2)*self.fs, spectrum.shape)
                figure.plot(f, spectrum)
                figure.set_xlabel('frequency [Hz]')
        figure.set_title(self.type + ' filter bank')


class GaussianFilter(Filter):
    """Gaussian filter.

    Generates a (modulated) Gaussian filter, parameterized via its central frequency and sigma.
    """
    # parameters: mean - frequency and deviation - window size
    # skewness and kurtosis can be incorporated later (PARAMETER_NUMBER = 4 in that case)
    PARAMETER_NUMBER = 2
    PARAMETER_NAMES = ['Frequency', 'Sigma']
    COSINE_AND_SINE_FILTER = False

    def __init__(self, fs, filter_length=500):
        self.fs = fs
        self.filter_length = filter_length
        self.type = 'Gaussian'
        super(GaussianFilter, self).__init__(fs, self.type, self.COSINE_AND_SINE_FILTER,
        self.PARAMETER_NUMBER, self.PARAMETER_NAMES)
        
    def __repr__(self):
        return 'Filter type: %s' % (self.type) 
        
    def get_kernel(self, parameters):
        n = (np.arange(0, self.filter_length) - (self.filter_length-1.0)/2).astype('float32')
        t = n
        t2 = t**2
        cos_win = tf.cos(parameters[0]*2*np.pi*t)
        gaus_win = tf.sqrt( 2 / ( np.sqrt(np.pi) * tf.keras.backend.variable(parameters[1]) ) )
        gaus_win = gaus_win * tf.exp(-t2/(2*parameters[1]**2)) 
        kern = gaus_win * cos_win
        kern = tf.reshape(kern, (len(t), 1))
        return kern
    
    def init_filter_bank(self, shape):
        # we want to learn mean and deviation on a range 0-0.5
        nb_filters = shape[0]
        mu = np.linspace(0, 0.5, nb_filters, endpoint=False)
        sigma = nb_filters #self.filter_length/40
        print('Sigma for Gaussian: ', sigma)
        sigma = np.multiply(np.ones(mu.shape), sigma)
        x = np.vstack((mu, sigma)).transpose()
        return x


class GaussianFilterCS(Filter):
    """Gaussian complex filter

    Generates a (cosine+sine modulated) Gaussian filter, parameterized via its central frequency and sigma.
    """
    # parameters: mean - frequency and deviation - window size
    # skewness and kurtosis can be incorporated later (PARAMETER_NUMBER = 4 in that case)
    PARAMETER_NUMBER = 2
    PARAMETER_NAMES = ['Frequency', 'Sigma']
    COSINE_AND_SINE_FILTER = True
    def __init__(self, fs, filter_length=500):
        self.fs = fs
        self.filter_length = filter_length
        self.type = 'GaussianCS'
        super(GaussianFilterCS, self).__init__(fs, self.type, self.COSINE_AND_SINE_FILTER,
        self.PARAMETER_NUMBER, self.PARAMETER_NAMES)
             
    def __repr__(self):
        return 'Filter type: %s' % (self.type) 
        
    def get_kernel(self, parameters):
        n = (np.arange(0, self.filter_length) - (self.filter_length-1.0)/2).astype('float32')
        t = n
        t2 = t**2
        freq = tf.abs(parameters[0])
        sigma = parameters[1]
        cos_win = tf.cos(freq*2*np.pi*t)
        sin_win = tf.sin(freq*2*np.pi*t)
        gaus_win = tf.sqrt(2/(np.sqrt(np.pi) * sigma))
        gaus_win = gaus_win * tf.exp(-t2/(2*sigma*sigma)) 
        kern_cos = gaus_win * cos_win
        kern_cos = tf.reshape(kern_cos, (len(t), 1))
        kern_sin = gaus_win * sin_win
        kern_sin = tf.reshape(kern_sin, (len(t), 1))
        return [kern_cos,kern_sin]
    
    def init_filter_bank(self, shape):
        # we want to learn mean and deviation on a range 0-0.5
        nb_filters = shape[0]
        mu = np.linspace(0.01, 0.5, nb_filters, endpoint=False)
        sigma = nb_filters #self.filter_length/40
        print('Sigma for Gaussian: ', sigma)
        sigma = np.multiply(np.ones(mu.shape), sigma)
        x = np.vstack((mu, sigma)).transpose()
        return x
        

     
class WaveletFilter(Filter):
    """Wavelet filter.

    From the paper "Learning filter widths of spectral decompositions with wavelets." NeurIPS 2018.
    cf https://github.com/haidark/WaveletDeconv    
    """
    # parameters: s - scale
    PARAMETER_NUMBER = 1
    PARAMETER_NAMES = ['Scale']
    COSINE_AND_SINE_FILTER = False
    def __init__(self, fs=16000, filter_length=500):
        self.fs = fs
        self.filter_length = filter_length
        self.type = 'Wavelet'
        super(WaveletFilter, self).__init__(fs, self.type, self.COSINE_AND_SINE_FILTER, self.PARAMETER_NUMBER, self.PARAMETER_NAMES) 
        
    def __repr__(self):
        return 'Filter type: %s' % (self.type) 
    
    def get_kernel(self, parameters):
        n = (np.arange(0, self.filter_length) - (self.filter_length-1.0)/2).astype('float32')
        t = n
        t2 = t**2
        scale = parameters
        scale2 = scale**2
        B = (3 * scale)**0.5
        A = (2 / (B * (np.pi**0.25)))
        mod = (1 - (t2)/(scale2))
        gauss = tf.exp(-(t2) / (2 * (scale2)))
        kern = A * mod * gauss
        kern = tf.reshape(kern, (len(t), 1))
        return kern/tf.norm(kern)
   
    def init_filter_bank(self, shape,):
        nb_filters = shape[0]
        x = np.logspace(0, 1.4, nb_filters, endpoint=True)
        x = np.reshape(x,(len(x), 1))
        return x


class GammatoneFilter_o4(Filter):
    """Real-valued fixed order (4) Gammatone filter

    Generates an order 4 Gammatone filter, parameterized via its central frequency and bandwidth.
    """
    # parameters:  b - bandwidth, f - frequency
    PARAMETER_NUMBER = 2
    PARAMETER_NAMES = ['Frequency','b (decay)']
    COSINE_AND_SINE_FILTER = False
    def __init__(self, fs=16000, filter_length=500):
        self.fs = fs
        self.filter_length = filter_length
        self.type = 'Gammatone fixed order=4'
        super(GammatoneFilter_o4, self).__init__(fs, self.type, self.COSINE_AND_SINE_FILTER, self.PARAMETER_NUMBER, self.PARAMETER_NAMES)  
        
    def __repr__(self):
        return 'Filter type: %s' % (self.type) 
    
    def get_kernel(self, parameters):
        f = parameters[0]
        b = parameters[1]
        order = 4.0
        
        n = (np.arange(0, self.filter_length)).astype('float64')
        n = np.reshape(n, (len(n), 1))
        t = n#/self.fs
        
        alpha = order - 1
        gtone = t**(alpha)*tf.exp(-2*np.pi*b*t)*tf.cos(2*np.pi*f*t)
        gnorm = (4*np.pi*b)**((2*alpha+1)/2)/tf.sqrt(tf.exp(tf.math.lgamma(2*alpha+1)))*np.sqrt(2) # gamma function G(2*n+1)
        gtone = gtone * gnorm
        return gtone
    
    def init_filter_bank(self, shape):
        nb_filters = shape[0]
        b = np.arange(0.03, 0.1, 0.01)
        f = np.linspace(0, 0.5, nb_filters, endpoint=False)
        index = 0
        x = np.zeros((self.PARAMETER_NUMBER, nb_filters))
        for index in range(nb_filters):
            x[0,index] = f[index]
            x[1,index] = float(1/nb_filters/4)
            
        x = x.transpose()
        return x
    
class GammatoneFilter(Filter):
    """Gammatone filter

    Generates an Gammatone filter, parameterized via its order, central frequency and bandwidth.
    """
    PARAMETER_NUMBER = 3
    PARAMETER_NAMES = ['Frequency','b (decay)','Order']
    COSINE_AND_SINE_FILTER = False
    def __init__(self, fs=16000, filter_length=500):
        self.fs = fs
        self.filter_length = filter_length
        self.type = 'Gammatone'
        super(GammatoneFilter, self).__init__(fs, self.type, self.COSINE_AND_SINE_FILTER,
        self.PARAMETER_NUMBER, self.PARAMETER_NAMES)  
        
    def __repr__(self):
        return 'Filter type: %s' % (self.type) 
    
    def get_kernel(self, parameters):
        f = parameters[0]
        b = parameters[1]
        order = parameters[2]
        
        n = (np.arange(0, self.filter_length)).astype('float64')
        n = np.reshape(n, (len(n), 1))
        t = n#/self.fs
        
        alpha = order - 1
        gtone = t**(alpha)*tf.exp(-2*np.pi*b*t)*tf.cos(2*np.pi*f*t)
        gnorm = (4*np.pi*b)**((2*alpha+1)/2)/tf.sqrt(tf.exp(tf.math.lgamma(2*alpha+1)))*np.sqrt(2) # gamma function G(2*n+1)
        gtone = gtone * gnorm
        return gtone
    
    def init_filter_bank(self, shape):
        nb_filters = shape[0]
        b = np.arange(0.03, 0.1, 0.01)
        f = np.linspace(0, 0.5, nb_filters, endpoint=False)
        index = 0
        x = np.zeros((self.PARAMETER_NUMBER, nb_filters))
        for index in range(nb_filters):
            x[0,index] = f[index]
            x[1,index] = float(1/nb_filters/4)
            x[2,index] = 4.0 # order starts at 2
        x = x.transpose()
        return x

class GammatoneFilterCS(Filter):
    """Complex Gammatone filter.

    Generates a complex Gammatone filter, parameterized via its order, central frequency, bandwidth and optionally amplitude.
    """
    # parameters: amplitude, order, b - bandwidth, f - frequency
    # Return a couple of functions one with cos and one with sin modulation

    COSINE_AND_SINE_FILTER = True
    def __init__(self, fs=16000, filter_length=500, order=4, fixed_amplitude=True):
        self.PARAMETER_NUMBER = 4
        self.PARAMETER_NAMES = ['Frequency','b (decay)','Order','Amplitude']        
        self.fs = fs
        self.filter_length = filter_length
        self.type = 'GammatoneCS'
        self.order = np.float32(order)
        self.fixed_amplitude = fixed_amplitude
        if order is not None:
            self.PARAMETER_NUMBER -= 1
            self.PARAMETER_NAMES.pop(self.PARAMETER_NAMES.index('Order'))
            self.type += '_o' + str(order)

        if fixed_amplitude:
            self.PARAMETER_NUMBER -=1
            self.PARAMETER_NAMES.pop(self.PARAMETER_NAMES.index('Amplitude'))
            self.type += ' (fixed amplitude)'

        super(GammatoneFilterCS, self).__init__(fs, self.type, self.COSINE_AND_SINE_FILTER,
        self.PARAMETER_NUMBER, self.PARAMETER_NAMES) 
        
    def __repr__(self):
        return 'Filter type: %s' % (self.type) 
    
    def get_kernel(self, parameters):
        f = parameters[0]            
        b = parameters[1] 
        if self.PARAMETER_NUMBER == 4:
            order = parameters[2]
            amplitude = parameters[3]
        elif self.PARAMETER_NUMBER == 3:
            if self.fixed_amplitude:
                order = parameters[2]
                amplitude = 1
            else:
                amplitude = parameters[2]
                order = self.order
        else:
            amplitude = 1
            order = self.order
            
        n = (np.arange(0, self.filter_length))
        n = np.reshape(n, (len(n), 1))
        t = n
        alpha = order - 1
        modulation_cos = tf.cos(2*np.pi*f*t)
        modulation_sin = tf.sin(2*np.pi*f*t)
        envelop = t**(alpha)*tf.exp(-2*np.pi*b*t) 
        gnorm1 = (4*np.pi*b)**((2*alpha+1)/2)
        gnorm2 = tf.cast(tf.sqrt(tf.exp(tf.math.lgamma(2*alpha+1)))*np.sqrt(2),dtype='float32') # gamma function G(2*n+1)
        gnorm = gnorm1 / gnorm2
        gtoneC = envelop*modulation_cos*gnorm*amplitude
        gtoneS = envelop*modulation_sin*gnorm*amplitude
        return [gtoneC,gtoneS]
    
    def init_filter_bank(self, shape):
        nb_filters = shape[0]
        f = np.linspace(0, 0.5, nb_filters, endpoint=False)
        index = 0
        import random
        x = np.zeros((self.PARAMETER_NUMBER, nb_filters))
        for index in range(nb_filters):   
            x[0,index] = random.choice(np.arange(10,100)*0.005)#f[index]
            x[1,index] = random.choice(np.arange(10,100)*0.001)#float(1/nb_filters/4)#np.random.choice(b)#0.04  
            if self.PARAMETER_NUMBER == 4:
                x[2,index] = 2.0 # order
                x[3,index] = 1.0 # amplitude
            elif self.PARAMETER_NUMBER == 3:
                if self.fixed_amplitude:
                    x[2,index] = 2.0 # order
                else:
                    x[2,index] = 1.0 # amplitude
        x = x.transpose()
        return x

#####################################################
class GammachirpFilterCS(Filter):
    """Complex Gammachirp filter

    Generates a complex Gammachirp filter of fixed order (4), parameterized using its central frequency, bandwidth and chirp parameter.
    """
    PARAMETER_NUMBER = 3
    PARAMETER_NAMES = ['Frequency','b (decay)','c (log slope)']
    COSINE_AND_SINE_FILTER = True
    def __init__(self, fs=16000, filter_length=500):
        self.fs = fs
        self.filter_length = filter_length
        self.type = 'Gammachirp_CS'
        super(GammachirpFilterCS, self).__init__(fs, self.type, self.COSINE_AND_SINE_FILTER,
        self.PARAMETER_NUMBER, self.PARAMETER_NAMES) 
        
    def __repr__(self):
        return 'Filter type: %s' % (self.type) 
    
    def get_kernel(self, parameters):
        f = parameters[0]        
        b = parameters[1]
        c = parameters[2]
        n = (np.arange(0, self.filter_length)).astype('float64')
        n = np.reshape(n, (len(n), 1))
        t = n
        g = 1 # Gammatone of order 4
        order = 2*(g+1)
        alpha = order -1
        eps = 0.0001
        modulation_cos = tf.cos(2*np.pi*f*t + c * np.log(t + eps) )
        modulation_sin = tf.sin(2*np.pi*f*t + c * np.log(t + eps) )
        envelop = t**(alpha)*tf.exp(-2*np.pi*b*t)
        gnorm = (4*np.pi*b)**((2*alpha+1)/2)/np.sqrt(scipy.special.gamma(2*alpha+1))*np.sqrt(2) # gamma function G(2*n+1)       
        gtoneC = envelop*modulation_cos*gnorm
        gtoneS = envelop*modulation_sin*gnorm
        return [gtoneC,gtoneS]
    
    def init_filter_bank(self,shape):
        nb_filters = shape[0]
        f = np.linspace(0.005, 0.5, nb_filters, endpoint=False)
        x = np.zeros((self.PARAMETER_NUMBER, nb_filters))
        
        for index in range(nb_filters):
            x[0,index] = f[index]
            x[1,index] = 1/nb_filters/4
            x[2,index] = 0.0
        x = x.transpose()
        return x
    

class SincFilter(Filter):
    """Bandpass filter from 2 sinc.

    From "Speaker Recognition from Raw Waveform with SincNet", https://arxiv.org/abs/1808.00158
    https://github.com/mravanelli/SincNet

    Generates a bandpass filter parameterized using lower and upper frequencies.
    """
    PARAMETER_NUMBER = 2
    PARAMETER_NAMES = ['f_min', 'f_max']
    COSINE_AND_SINE_FILTER = False

    def __init__(self, fs, filter_length=500, use_mel_init=True):
        self.fs = fs
        self.filter_length = filter_length
        self.type = 'Sinc'
        self.use_mel_init = use_mel_init
        
        if(fs == 8000):
            self.cw_len=375
            self.cw_shift=10
        else:          
            self.cw_len=200
            self.cw_shift=10
        super(SincFilter, self).__init__(fs, self.type, self.COSINE_AND_SINE_FILTER,
        self.PARAMETER_NUMBER, self.PARAMETER_NAMES)
        
    def __repr__(self):
        return 'Filter type: %s' % (self.type)
        
    def get_kernel(self, parameters):
        n = np.arange(0, self.filter_length)
        t = (n - (self.filter_length-1.0)/2).astype('float32')
        
        low_pass1 = 2*parameters[0]*tf.sin(2*np.pi*parameters[0]*t)/ \
        (2*np.pi*parameters[0]*t)   
        low_pass2 = 2*parameters[1]*tf.sin(2*np.pi*parameters[1]*t)/ \
        (2*np.pi*parameters[1]*t)        
        
        band_pass = (low_pass2 - low_pass1)
        
        window = 0.54 - 0.46*np.cos(2*np.pi*n/self.filter_length)
        kern = tf.reshape(band_pass*window, (len(t), 1))
        return kern
    
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)
    
    def init_filter_bank_lin(self, shape):
        nb_filters = shape[0]
        #b = np.arange(0.03, 0.1, 0.01)
        f = np.linspace(0, 0.5, nb_filters, endpoint=False)#*self.fs # CHANGED
        index = 0
        x = np.zeros((self.PARAMETER_NUMBER, nb_filters))
        for index in range(nb_filters):
            x[0,index] = f[index]
            x[1,index] = f[index] + 0.25*nb_filters
            
        x = x.transpose()
        return x
    
    def init_filter_bank_mel(self, shape):
        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        min_low_hz = 50
        min_band_hz = 50
        nb_filters = shape[0]
        high_hz = self.fs/2 - (min_low_hz + min_band_hz)
        
        mel_low = 2595 * np.log10(1 + low_hz / 700)
        mel_high = 2595 * np.log10(1 + high_hz / 700)
        
        mel = np.linspace(mel_low, mel_high, nb_filters + 1)
        hz_low = 700 * (10 ** (mel / 2595) - 1)
        
        band_hz = np.diff(hz_low)
        hz_low = hz_low[:-1]
        hz_high = hz_low + min_band_hz + min_low_hz
        hz_high = np.clip(hz_high + np.abs(band_hz), min_low_hz, self.fs/2)
        x = np.vstack((hz_low, hz_high)).transpose()
        x = x/self.fs
        return x
    
    def init_filter_bank(self, shape):
        if self.use_mel_init:
            return self.init_filter_bank_mel(shape)
        else:
            return self.init_filter_bank_lin(shape)
        
FILTER_NAMES = ['Wavelet', 'GaussianCS', 'GammatoneCS', 'GammachirpCS', 'Gammatone', 'Gammatone_o4', 'SincNet', 'SincNet-NoMel']
NUM_FILTER_TYPES = 8
def create_filter_layer(filt_type, fs, filter_length):
    if filt_type == 0:
        return WaveletFilter(fs, filter_length)
    elif filt_type == 1:
        return GaussianFilterCS(fs, filter_length)
    elif filt_type == 2:
        return GammatoneFilterCS(fs, filter_length) # order = 4, fixed_amplitude=True by default
    elif filt_type == 3:
        return GammachirpFilterCS(fs, filter_length)
    elif filt_type == 4:
        return GammatoneFilter(fs, filter_length)
    elif filt_type == 5:
        return GammatoneFilter_o4(fs, filter_length)
    elif filt_type == 6:
        return SincFilter(fs, filter_length)
    elif filt_type == 7:
        return SincFilter(fs, filter_length, False)
    raise Exception('Unknown filter type')
