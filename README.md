This repository contains the code to reproduce the experiments from the [paper](https://septentrio.uit.no/index.php/nldl/article/view/6279) "Learnable filter-banks for CNN-based audio applications" published at [NLDL'2022](https://www.nldl.org)

This code has been written by Benjamin Ricaud[^1][^2], Helena Peic Tukuljac[^2], Nicolas Aspert[^2] and Laurent Colbois[^3]. If you use it, please cite the paper accordingly:
```
@inproceedings{PeicTukuljac:293274,
      title = {Learnable filter-banks for CNN-based audio applications},
      author = {Peic Tukuljac, Helena and Ricaud, Benjamin and Aspert,  Nicolas and Colbois, Laurent},
      journal = {Proceedings of the Northern Lights Deep Learning Workshop  2022 },
      series = {Proceedings of the Northern Lights Deep Learning Workshop.  3},
      pages = {9},
      year = {2022},
      abstract = {We investigate the design of a convolutional layer where  kernels are parameterized functions. This layer aims at  being the input layer of convolutional neural networks for  audio applications or applications involving time-series.  The kernels are defined as one-dimensional functions having  a band-pass filter shape, with a limited number of  trainable parameters. Building on the literature on this  topic, we confirm that networks having such an input layer  can achieve state-of-the-art accuracy on several audio  classification tasks. We explore the effect of different  parameters on the network accuracy and learning ability.  This approach reduces the number of weights to be trained  and enables larger kernel sizes, an advantage for audio  applications. Furthermore, the learned filters bring  additional interpretability and a better understanding of  the audio properties exploited by the network.},
      url = {https://septentrio.uit.no/index.php/nldl/article/view/6279},
      doi = {10.7557/18.6279},
}
```

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
