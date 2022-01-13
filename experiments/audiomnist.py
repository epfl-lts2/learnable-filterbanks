import logging
import gin
import h5py
import argparse
import json
import numpy as np
import experiments.models
import tensorflow as tf
from tensorflow.keras.utils import to_categorical




@gin.configurable
def train_test_split(train_data, train_labels, validation_data, validation_labels, 
                test_data, test_labels,
                model=None, num_epochs=50, batch_size=128):
    """Perform training of the model and evaluation on test data.

    Args:
        train_data ([type]): [description]
        train_labels ([type]): [description]
        validation_data ([type]): [description]
        validation_labels ([type]): [description]
        test_data ([type]): [description]
        test_labels ([type]): [description]
        model ([type], optional): [description]. Defaults to None.
        num_epochs (int, optional): [description]. Defaults to 50.
        batch_size (int, optional): [description]. Defaults to 128.

    Returns:
        [type]: [description]
    """
    model.summary()
    history = model.fit(train_data, train_labels, batch_size=batch_size, epochs=num_epochs, 
            validation_data=(validation_data, validation_labels))
    res = model.evaluate(test_data, test_labels)
    return model, history, res


@gin.configurable
def load_split(filename, reshape_data=False, data_format='channels_first'):
    with h5py.File(filename, 'r') as f:
        train_data = f['data/train'][:]
        train_labels = to_categorical(f['label/train'][:])
        validation_data = f['data/validate'][:]
        validation_labels = to_categorical(f['label/validate'][:])
        test_data = f['data/test'][:]
        test_labels = to_categorical(f['label/test'][:])
        if reshape_data:
            if data_format == 'channels_first':
                train_data = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1]))
                validation_data = np.reshape(validation_data, (validation_data.shape[0], 1, validation_data.shape[1]))
                test_data = np.reshape(test_data, (test_data.shape[0], 1, test_data.shape[1]))
            elif data_format == 'channels_last':
                train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))
                validation_data = np.reshape(validation_data, (validation_data.shape[0], validation_data.shape[1], 1))
                test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))
            else:
                raise ValueError('Unknown data_format ', data_format)
        print(train_data.shape)
        return (train_data, train_labels), (validation_data, validation_labels), (test_data, test_labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='gin configuration file')
    parser.add_argument('--split-file', help='HDF5 file containing the train/validation/test split of AudioMNIST')
    parser.add_argument('--result-output', help='Write results (training history + evaluation) to the specified file (json format)')
    parser.add_argument('--model-output', help='Write the trained model to the specified file')
    args = parser.parse_args()
    gin.parse_config_file(args.config)
    train, val, test = load_split(args.split_file)
    model, history, res = train_test_split(train[0], train[1], val[0], val[1], test[0], test[1])
    if args.result_output:
        with open(args.result_output, 'w') as f:
            json.dump({'history': history.history, 'result': res}, f)
    if args.model_output:
        model.save(args.model_output)
    

if __name__ == '__main__':
    # set the following environment variables before running if you want to reduce TF verbosity
    # export TF_CPP_MIN_LOG_LEVEL=2
    # export AUTOGRAPH_VERBOSITY=0
    tf.get_logger().setLevel(logging.DEBUG)
    main()
