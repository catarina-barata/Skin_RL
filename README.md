# 	Incorporating human values into AI-based decision support
 
### Overview
This repository contains the source code to implement a deep-Q learning model that incorporates human-values in the diagnostic process. The model receives features from a standard neural network (e.g. CNN) as well as the softmax probabilities for the different classes. It then makes the final decision, based on the policy learned from a reward table defined by medical experts. The model may keep or change the diagnosis when compared witht the CNN. Additionally, we consider the possibility for the model to pick an "unknown" action, where no class is picked.

### System Requirements
Most experiments were run using a desktop with 16Gb of RAM, an Intel i5-7600 CPU @350HZ, and an NIVIDA Titan Xp. 
However, given the type of data and the characterisitcs of the deep Q-learning model, our demo can be easily run in a regular desktop computer without GPU. The user must only guarantee a minimum of 16Gb of RAM to accomodate both model and data.

The models were trained on Microsoft Windows 10 Pro 64bit and Ubunto 18.04

### Installation
To run the demo, it is necessary to install the following python packages:
* Python >= 3.8
* tensorflow==2.8.0
* numpy
* scikit-learn
* pandas

 
