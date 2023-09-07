# 	A reinforcement learning model for AI-based decision support in skin cancer
 
### Overview
This repository contains the source code to implement a deep-Q learning model that incorporates human-values in the diagnostic/management process of melanoma patients. The model receives features from a standard neural network (e.g. CNN) as well as the softmax probabilities for the different classes. It then makes a recommendation, based on the policy learned from a reward table defined by medical experts. The model may keep or change the diagnosis when compared witht the CNN. Additionally, we consider the possibility for the model to recommend management actions.

### System Requirements
Most experiments were run using a desktop with 16Gb of RAM, an Intel i5-7600 CPU @350HZ, and an NIVIDA Titan Xp. 
However, given the type of data and the characterisitcs of the deep Q-learning model, our demo can be easily run in a regular desktop computer without GPU. The user must only guarantee a minimum of 16Gb of RAM to accomodate both model and data.

The models were trained on Microsoft Windows 10 Pro 64bit and Ubunto 18.04

### Installation
* To install the required packages: pip install -r requirements.txt
* To install the demo: Follow https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository and use the following link https://github.com/catarina-barata/Skin_RL.git

### Data preparation
* Download the datasets to the **data** folder, as described in the data/Data_Download.txt file

### Run Demo - RL for Diagnosis
To run the demo (train an RL agent and make predictions on a validation set):
1) Go  to the corresponding directory

2) Run the python RL_Skin_Cancer_Demo_Diagnosis.py

3) You can manipulate the **number of patients per episode** (episode length/number of iterations), the **number of episodes**, and whether to **use or not the unknown action**.
An example: python RL_Skin_Cancer_Demo_Diagnosis.py --n_patients 100 --n_episodes 150 --use_unknown False

4) We give the possibility to try with different reward tables, where the penalty for the use of the **Unknown** class is changed - simply (un)comment the code of the desired reward in funtion **step** (class **Dermatologist**).
You can also use this function to define new reward tables

### Run Demo - RL for lesion/image-level management
To run the demo (train an RL agent and make predictions on a validation set):
1) Go  to the corresponding directory

2) Run the python RL_Skin_Cancer_Demo_Management.py

3) You can manipulate the **number of patients per episode** (episode length/number of iterations), the **number of episodes**, and the **number of actions** (2 - dismiss/excise or 3 dismiss/treat locally/excise).
An example: python RL_Skin_Cancer_Demo_Management.py --n_patients 100 --n_episodes 150 --n_actions 2

4) We give the possibility to try with different expert reward tables for the 2 actions problem - simply (un)comment the code of the desired reward in funtion **step** (class **Dermatologist**).
You can also use this function to define new reward tables

### Run Demo - RL for patient-level management
To run the demo (train an RL agent and make predictions on a validation set):
1) Go  to the corresponding directory

2) Run the python Skin_Cancer_RL_Demo_Patient_Management.py

3) You can manipulate the **number of patients seen before updating Q-network**, the **number of macro episodes** - i.e. the number of times the full dataset is run  and the **number of actions** (2 - dismiss/excise or 3 dismiss/monitor/excise).
An example: python Skin_Cancer_RL_Demo_Patient_Management.py --n_patients 1 --n_episodes 130 --n_actions 3

**Expected Outcome** - You should be able to train a RL models that are able to predict the Q-values of the different actions (diagnostic/management decisions) given the features and softmax probabilities of a standard supervised model. The model then chooses the action with the highest Q-value. A numerical evaluation is carried out using the confusion matrix to show the performance of the RL model on a validation set.

### Try your data
To try new data, some modifications must be done:
1) If using the same 7 classes of skin lesions, but different images and/or different CNN - you just need to save the features into a **numpy array** and the probabilities, image id, and real diagnosis into a **CSV**. Please check the formats used in the demo examples (data folder). If working with the patient-level setting, you will need to creat a numpy matrix with the following structure (n_patients x n_images x features) and the corresponding ground truth diagnosis - see the provided examples **patient_embeddings.npy** and **gt_patients_embeddings**.

2) For new (medical) problems, you will also need to adjust the **initialize_clinical_practice** to your dataset, as well as the reward tables.
You may also need to adjust the Q-network with additional layers (function **create_q_model**).

### Reference

@article{barata2023reinforcement,
  title={A reinforcement learning model for AI-based decision support in skin cancer},
  author={Barata, Catarina and Rotemberg, Veronica and Codella, Noel CF and Tschandl, Philipp and Rinner, Christoph and Akay, Bengu Nisa and Apalla, Zoe and Argenziano, Giuseppe and Halpern, Allan and Lallas, Aimilios and others},
  journal={Nature Medicine},
  pages={1--6},
  year={2023},
  publisher={Nature Publishing Group US New York}
}
 
