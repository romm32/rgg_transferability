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
@INPROCEEDINGS{11460522,
  author={Camargo, Romina Garcia and Wang, Zhiyang and Ribeiro, Alejandro},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Graph Neural Networks in Large Scale Wireless Communication Networks: Scalability Across Random Geometric Graphs}, 
  year={2026},
  volume={},
  number={},
  pages={616-620},
  keywords={Radio broadcasting;Frequency modulation;Filtering;Filters;Circuits and systems;Wireless communication;Wireless networks;Communication networks;Communication systems;Network architecture;transferability;graph neural networks;random geometric graphs},
  doi={10.1109/ICASSP55912.2026.11460522}}

The citation for the conference version will be added soon.

