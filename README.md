# 	Incorporating human values into AI-based decision support
 
### Overview
This repository contains the source code to implement a deep-Q learning model that incorporates human-values in the diagnostic process. The model receives features from a standard neural network (e.g. CNN) as well as the softmax probabilities for the different classes. It then makes the final decision, based on the policy learned from a reward table defined by medical experts. The model may keep or change the diagnosis when compared witht the CNN. Additionally, we consider the possibility for the model to pick an "unknown" action, where no class is picked.

### System Requirements
Most experiments were run using a desktop with 16Gb of RAM, an Intel i5-7600 CPU @350HZ, and an NIVIDA Titan Xp. 
However, given the type of data and the characterisitcs of the deep Q-learning model, our demo can be easily run in a regular desktop computer without GPU. The user must only guarantee a minimum of 16Gb of RAM to accomodate both model and data.

The models were trained on Microsoft Windows 10 Pro 64bit and Ubunto 18.04

### Installation
* To install the required packages: pip install -r requirements.txt
* To install the demo: Follow https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository and use the following link https://github.com/catarina-barata/Skin_RL.git

### Run Demo
To run the demo (train an RL agent and make predictions on a validation set):
1) Go  to the corresponding directory

2) Run python RL_Skin_Cancer_Demo.py

3) You can manipulate the **number of patients per episode** (episode length/number of iterations), the **number of episodes**, and whether to **use or not the unknown action**.
An example: - python RL_Skin_Cancer_Demo.py --n_patients 100 --n_episodes 150 --use_unknown False

4) We give the possibility to try with different reward tables, where the penalty for the use of the **Unknown** class is changed -simply (un)comment the code of the desired reward in funtion **step** (class **Dermatologist**)
 
