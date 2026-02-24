# rgg_transferability

Repository for the ICASSP paper "GRAPH NEURAL NETWORKS IN LARGE SCALE WIRELESS COMMUNICATION NETWORKS: SCALABILITY ACROSS RANDOM GEOMETRIC GRAPHS".

### Installation
You can clone the repository as is usually done:

_git clone https://github.com/romm32/rgg_transferability.git_

We provide a .yml file to set up a conda environment in Ubuntu 22, with which the installation of the packages should become easier.

### Use
The file data_generation enables generating a dataset. After this, you can run the main file inside the conda environment as follows.

_python main.py_

You can also specify training/evaluation parameters as arguments. You can request help via an email to _rominag@seas.upenn.edu_.

Please cite the papers if you use the code:

````bibtex
@misc{camargo2025graphneuralnetworkslarge,
      title={Graph Neural Networks in Large Scale Wireless Communication Networks: Scalability Across Random Geometric Graphs}, 
      author={Romina Garcia Camargo and Zhiyang Wang and Alejandro Ribeiro},
      year={2025},
      eprint={2510.00896},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2510.00896}, 
}

The citation for the conference version will be added soon.

