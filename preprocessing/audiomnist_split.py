import h5py
import json
import numpy as np
import os
from tqdm import tqdm
import argparse
import sys



def load_split_data(file_list, task_id):
    labels = []
    data = []
    for p in tqdm(file_list):
        with h5py.File(p, 'r') as f:
            lbl = f['label'][:]
            data.append(np.reshape(f['data'][:], (8000, 1)))
            labels.append(lbl[0][task_id])
    return np.array(data), np.array(labels)

def load_split(mnist_path, split_id):
    task_id = 0
    task = 'digit'
    file_base = '{}/AudioNet_{}_{}'.format(mnist_path, task, split_id)
    sub = ['train', 'test', 'validate']
    split_data = {}
    split_labels = {}
    for v in sub:
        file_list_path = '{}_{}.txt'.format(file_base, v)
        with open(file_list_path) as f:
            lines = f.readlines()
        lines = [p.rstrip('\n') for p in lines]
        data,labels = load_split_data(lines, task_id)
        split_data[v] = data
        split_labels[v] = labels
    return split_data, split_labels


def preprocess_mnist():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mnist-dir', help='Base directory of AudioMNIST repository clone')
    parser.add_argument('--output-dir', help='Output directory')
    args = parser.parse_args()
    mnist_dir = args.mnist_dir
    output_dir = args.output_dir

    root_folder = os.path.join(mnist_dir, 'preprocessed_data')
    if not os.path.isdir(root_folder):
        print('Unable to read AudioMNIST preprocessed data. (Did you forget to run AudioMNIST\'s own preprocessing script ?)')
        sys.exit()

    data_type = ['train', 'test', 'validate']
   
    for split_id in range(5):
        print('Processing split {}'.format(split_id))
        data, labels = load_split(root_folder, split_id)
        outputname = os.path.join(output_dir, 'audiomnist_split_{}.hdf5'.format(split_id))
        with h5py.File(outputname, 'w') as f:
            for t in data_type:
                f['data/{}'.format(t)] = data[t]
                f['label/{}'.format(t)] = labels[t]
            f['fs'] = np.array([8000]) #  Sampling frequency
            f['num_classes'] = np.array([10])


if __name__ == '__main__':
    preprocess_mnist()
