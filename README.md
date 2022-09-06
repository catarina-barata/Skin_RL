# 	Incorporating human values into AI-based decision support
 
### Overview
This repository contains the source code to implement a deep-Q learning model that incorporates human-values in the diagnostic process. The model receives features from a standard neural network (e.g. CNN) as well as the softmax probabilities for the different classes. It then makes the final decision, based on the policy learned from a reward table defined by medical experts. The model may keep or change the diagnosis when compared witht the CNN. Additionally, we consider the possibility for the model to pick an "unknown" action, where no class is picked.

### System Requirements
The model can be trained on a standard desktop computer, with 
