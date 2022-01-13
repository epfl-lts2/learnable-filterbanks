import argparse
import json
import gin
import functools
from typing import Dict, Mapping
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import tensorflow_datasets as tfds
import experiments.models as models

def db_to_linear(samples):
    return 10.0 ** (samples / 20.0)


def loudness_normalization(samples: tf.Tensor,
                           target_db: float = 15.0,
                           max_gain_db: float = 30.0):
    """Normalizes the loudness of the input signal."""
    std = tf.math.reduce_std(samples) + 1e-9
    gain = tf.minimum(db_to_linear(max_gain_db), db_to_linear(target_db) / std)
    return gain * samples


def align(samples: tf.Tensor, seq_len: int = 16000):
    pad_length = tf.maximum(seq_len - tf.size(samples), 0)
    return tf.image.random_crop(tf.pad(samples, [[0, pad_length]]), [seq_len])



def preprocess(inputs: Mapping[str, tf.Tensor],
               transform_fns=(align, loudness_normalization)):
    """Sequentially applies the transformations to the waveform."""
    audio = tf.cast(inputs['audio'], tf.float32) / tf.int16.max
    for transform_fn in transform_fns:
        audio = transform_fn(audio)
    return tf.expand_dims(audio, axis=0), inputs['label']

def prepare(datasets: Mapping[str, tf.data.Dataset],
            transform_fns=(align, loudness_normalization),
            batch_size: int = 64) -> Dict[str, tf.data.Dataset]:
    """Prepares the datasets for training and evaluation.
    Mostly duplicate from https://github.com/google-research/leaf-audio.
    """
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    valid = 'validation' if 'validation' in datasets else 'test'
    test = 'test' if 'test' in datasets else 'validation'
    if 'validation' not in datasets or 'test' not in datasets:
        print('Warning: Test and validation are identical.')
    result = {}
    for split, key in ('train', 'train'), (valid, 'eval'), (test, 'test'):
        ds = datasets[split]
        ds = ds.map(functools.partial(preprocess, transform_fns=transform_fns),
                num_parallel_calls=AUTOTUNE)
        result[key] = ds.batch(batch_size).prefetch(AUTOTUNE)
    return result

@gin.configurable
def train(data_dir=None, batch_size=128, num_epochs=50, num_filters=40, overlap=80, filter_length=400, filter_type=4,
          checkpoint_filepath=None, filters=[64, 128, 256, 256, 512, 512]):
    datasets = tfds.load('speech_commands', data_dir=data_dir)
    data_prep = prepare(datasets, batch_size=batch_size)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=4, min_lr=1e-4, verbose=1)
    callbacks = [reduce_lr]
    if checkpoint_filepath:
        model_checkpoint_cb = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        callbacks.append(model_checkpoint_cb)
    
    model = models.ConvNet(filters, 16000, 16000, num_filters=num_filters, overlap=overlap, 
                            filter_length=filter_length, num_classes=12, filter_type=filter_type)
    model.summary()
    history = model.fit(data_prep['train'], validation_data=data_prep['eval'], epochs=num_epochs, 
                        callbacks=callbacks, verbose=1)
    # reload best weight
    if checkpoint_filepath:
        model.load_weights(checkpoint_filepath)
    score = model.evaluate(data_prep['test'], batch_size=None, verbose=1)
    return model, history, score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='gin configuration file')
    parser.add_argument('--result-output', help='Write results (training history + evaluation) to the specified file (json format)')
    parser.add_argument('--model-output', help='Write the trained model to the specified file')
    args = parser.parse_args()
    gin.parse_config_file(args.config)
    model, history, res = train()
    if args.result_output:
        with open(args.result_output, 'w') as f:
            json.dump({'history': history.history, 'result': res}, f)
    if args.model_output:
        model.save(args.model_output)
    

if __name__ == '__main__':
    main()
