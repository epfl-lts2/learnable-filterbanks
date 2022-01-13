This repository contains the code to reproduce the experiments from the paper "Learnable filter-banks for CNN-based audio applications" published at [NLDL'2022](https://www.nldl.org)

This code has been written by Benjamin Ricaud[^1][^2], Helena Peic Tukuljac[^2], Nicolas Aspert[^2] and Laurent Colbois[^3].

# Installation
Create a virtual python environment (or conda) and install the requirements via `pip` (or `conda`):
```
pip install -r requirements.txt
```

# Experiments

There are two datasets used: [AudioMNIST](https://github.com/soerenab/AudioMNIST) and [Google Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands)

## AudioMNIST
* Clone the AudioMNIST repo
* Run the `audiomnist_split` script (located in the `preprocessing` directory)
* Adjust the `experiments/config_audiomnist.gin` to your needs
* Train a model using one of the preprocessed AudioMNIST splits:
```
python -m experiments.audiomnist  --config experiments/config_audiomnist.gin --split-file /data/AudioMNIST/audiomnist_split_0.hdf5 --result-output result_am0.json --model-output am_fb_an.h5
```

## Google Speech commands
* Adjust the `experiments/config_googlespeech.gin` to your needs
* Train a ConvNet model:
```
python -m experiments.google_speech --config experiments/config_googlespeech.gin --result-output result_gsc.json --model-output gsc.h5
```

---------
[^1]: Ecole Polytechnique Fédérale de Lausanne [LTS2](https://lts2.epfl.ch)

[^2]: Dept. of Physics and Technology, UiT The Arctic University of Norway, Tromsø

[^3]: IDIAP research center, Martigny
