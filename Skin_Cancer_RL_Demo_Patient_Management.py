from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import numpy as np
import math
import sklearn.metrics as metrics
from tqdm import tqdm
from gym import Env, spaces
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


from tensorflow.python.keras import backend
from tensorflow.keras.backend import clear_session


import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import csv

import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow.keras as K

from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

def read_and_decode(dataset, batch_size, is_training, data_size):
    if is_training:
        dataset = dataset.shuffle(buffer_size=data_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.repeat(None)
    else:
        dataset = dataset.prefetch(buffer_size=data_size // batch_size)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.repeat(None)
    return dataset

def create_q_model(n_classes,n_words,n_actions):

    ## MLP 
    inputs = K.layers.Input(n_classes)

    feat = K.layers.Lambda(lambda x: x[:, 0:n_classes - n_words])(inputs)

    emb = K.layers.Dropout(0.05)(feat)

    prob = K.layers.Lambda(lambda x: x[:, n_classes - n_words:n_classes])(inputs)

    emb = K.layers.Dense(256, activation="relu")(emb)

    emb = K.layers.Dropout(0.05)(emb)

    emb = K.layers.Concatenate(axis=1)([emb, prob])

    action = K.layers.Dense(n_actions, activation=None)(emb)

    return K.Model(inputs=inputs, outputs=action)

def initialize_clinical_practice(clinical_cases_feat,clinical_cases_labels,dataset_size,is_training):
    dataset_train = tf.data.Dataset.from_tensor_slices((clinical_cases_feat,clinical_cases_labels))

    dataset_train = read_and_decode(dataset_train, 1, is_training, dataset_size)

    patients = iter(dataset_train)

    return patients

def get_next_patient(patients):
    patient_scores,patient_diagnostics = patients.get_next()

    patient_scores = np.squeeze(patient_scores)

    posi = np.argsort(-patient_scores[:,516])

    patient_scores = patient_scores[posi,:]

    patient_diagnostics = patient_diagnostics.numpy()

    patient_diagnostics = patient_diagnostics[:,posi]

    return patient_scores,patient_diagnostics

class Dermatologist(Env):

    def __init__(self,patient_feat,patient_diag,n_classes,n_actions):
        # Actions we can take, either skin lesion classes or don't know
        self.action_space = spaces.Discrete(n_actions)
        # Observation space - softmax + features after GAP
        self.observation_space = spaces.Box(-1*math.inf*np.ones((n_classes,)),math.inf*np.ones((n_classes,)))

        # Initialize state
        n_state = np.squeeze(patient_feat[0,:])
        n_gt = int(np.squeeze(patient_diag)[0])
        self.state = n_state
        self.revised_state = self.state
        self.gt = n_gt

        # Set shower length
        self.number_of_cases = 1
        
        if np.any(np.where(np.sum(patient_feat,1)==0)) == False:
            self.limit = patient_feat.shape[0]-1
        else:
            self.limit = np.where(np.sum(patient_feat,1)==0)[0][0]
        
        self.context = np.mean(patient_feat[0:self.limit,0:512],axis=0)

        self.prob_context = np.mean(patient_feat[0:self.limit,512:512+n_classes],axis=0)
    
    def update_patient(self,patients,n_classes):
        
        pat_feat,pat_diag = get_next_patient(patients)
        
        n_state = np.squeeze(pat_feat[0,:])
        n_gt = int(np.squeeze(pat_diag)[0])
        self.state = n_state
        self.revised_state = self.state
        self.gt = n_gt

        # Set episode length
        self.number_of_cases = 1

        if np.any(np.where(np.sum(pat_feat,1)==0)) == False:

            self.limit = pat_feat.shape[0]-1
        else:
            self.limit = np.where(np.sum(pat_feat,1)==0)[0][0]
        
        self.context = np.mean(pat_feat[0:self.limit,0:512],axis=0)

        self.prob_context = np.mean(pat_feat[0:self.limit,512:512+n_classes],axis=0)
        
        return pat_feat,pat_diag

    def step(self,patient_feat,patient_diag,n_actions,action,next_id):
        ### CONSENSOUS -3 Actions
        reward_table = np.array([[ 5, -5], #dismiss
                                 [ 2, -1], #monitor
                                 [-1,  5], #excise
                                 ],np.float32)


        self.revised_state = tf.one_hot(action,n_actions)

        ### Usual reward
        reward = reward_table[action,int(self.gt)]

        n_state = np.squeeze(patient_feat[next_id,:])
        n_gt = int(np.squeeze(patient_diag)[next_id])

        old_gt = self.gt

        self.state = n_state
        self.gt = n_gt

        self.number_of_cases += 1

        # Check if checking patients is done
        if self.number_of_cases > self.limit:# or old_gt != action:
            done = 1
        else:
            done = 0
        
        return self.revised_state, self.state, reward,done,old_gt

    def reset(self,clinical_cases_feat,clinical_cases_labels,dataset_size,is_training,n_classes):
        # Reset clinical practice
        patients = initialize_clinical_practice(clinical_cases_feat,clinical_cases_labels, dataset_size,is_training)
        pat_feat,pat_diag = get_next_patient(patients)

        n_state = pat_feat[0,:]
        n_gt = np.squeeze(pat_diag)[0]

        self.state = n_state
        self.revised_state = self.state
        self.gt = n_gt

         # Set shower length
        self.number_of_cases = 1

        if np.any(np.where(np.sum(pat_feat,1)==0)) == False:

            self.limit = pat_feat.shape[0]-1
        else:
            self.limit = np.where(np.sum(pat_feat,1)==0)[0][0]
        
        self.context = np.mean(pat_feat[0:self.limit,0:512],axis=0)

        self.prob_context = np.mean(pat_feat[0:self.limit,512:512+n_classes],axis=0)

        return self.state,pat_feat,pat_diag,patients

def main(_):
    gamma = 0.99  # Discount factor for past rewards
    epsilon = 0.2  # Epsilon greedy parameter
    epsilon_min = 0.1  # Minimum epsilon greedy parameter
    epsilon_max = 0.2  # Maximum epsilon greedy parameter
    epsilon_interval = (epsilon_max - epsilon_min)  # Rate at which to reduce chance of random action being taken

    #### Import Datasets ####
    tf1.enable_eager_execution()

    vocab = ['ben','mel']

    n_words = 7

    features = np.load('data/patient_embeddings.npy')

    labels_cat = np.load('data/gt_patients_embeddings.npy')

    kf = KFold(n_splits=20,random_state=1111,shuffle = True)

    kf.get_n_splits(labels_cat)
    
    print(kf)

    for set, (train_index,test_index) in enumerate(kf.split(features,labels_cat)):

        train_feat = features[train_index,:,:]

        train_labels = labels_cat[train_index,:]

        val_feat = features[test_index,:,:]

        val_labels = labels_cat[test_index,:]

        patients = initialize_clinical_practice(train_feat,train_labels,train_labels.shape[0],True)

        pat_feat,pat_diag = get_next_patient(patients)

        print(pat_feat.shape)

        derm = Dermatologist(pat_feat,pat_diag,n_words,Flags.n_actions)

        q_network = create_q_model(derm.state.shape[0],n_words,Flags.n_actions)

        q_network.summary()

        target_network = create_q_model(derm.state.shape[0],n_words,Flags.n_actions)

        optimizer = K.optimizers.Adam(learning_rate=0.025, clipnorm=1.0)

        # Experience replay buffers
        action_history = []
        state_history = []
        state_next_history = []
        rewards_history = []
        done_history = []
        episode_reward_history = []
        episode_val_reward_history = []
        validation_bacc_history = []
        best_bacc = 0
        best_reward = -1*math.inf
        best_mel_reward = np.sum(val_labels)*(-5)
        best_ben_reward = -50
        iter_count = 0
        # Number of frames to take random action and observe output
        epsilon_random_frames = 20
        # Number of frames for exploration
        epsilon_greedy_frames = 100000.0
        # Maximum replay length
        max_memory_length = 10000
        # Train the model after 10 actions
        update_after_actions = 35#10
        # How often to update the target network
        update_target_network = 5800
        # Using huber loss for stability
        loss_function = K.losses.Huber()

        for macro_episode in range(Flags.n_episodes):
            print('Starting Macro Episode ',macro_episode)

            total_episode_score = 0

            with tqdm(total=train_feat.shape[0]) as pbar:
            
                for episode in range(train_feat.shape[0]):
                    i = 1
                    
                    #print('Starting Patient ',episode)

                    done = False
                    episode_score = 0

                    episode_val_score = 0

                    state = derm.state-np.concatenate([derm.context,derm.prob_context])

                    image_count = 1
                    n_not_random = 0
                    while not done:
                        try:
                            iter_count += 1

                            # Use epsilon-greedy for exploration
                            if iter_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                                action = derm.action_space.sample()
                            else:
                                state_tensor = tf.convert_to_tensor(state)
                                state_tensor = tf.expand_dims(state_tensor, 0)
                                action_probs = q_network(state_tensor, training=False)

                                # Take best action
                                action = tf.argmax(action_probs[0]).numpy()

                                n_not_random += 1

                            # Decay probability of taking random action
                            epsilon -= epsilon_interval / epsilon_greedy_frames
                            epsilon = max(epsilon, epsilon_min)

                            revised_state,n_state,reward,done,_ = derm.step(pat_feat,pat_diag,Flags.n_actions,action,image_count)

                            episode_score += reward

                            n_state = n_state-np.concatenate([derm.context,derm.prob_context])


                            # Save actions and states in replay buffer
                            action_history.append(action)
                            state_history.append(state)
                            state_next_history.append(n_state)
                            done_history.append(done)
                            rewards_history.append(reward)
                            state = n_state

                            i += 1
                            # Update every fourth frame
                            if iter_count % update_after_actions == 0 and len(done_history) > 100:
                                # Get indices of samples for replay buffers
                                indices = np.random.choice(range(len(done_history)), size=100)

                                # Using list comprehension to sample from replay buffer
                                state_sample = np.array([state_history[i] for i in indices])
                                state_next_sample = np.array([state_next_history[i] for i in indices])
                                rewards_sample = [rewards_history[i] for i in indices]
                                action_sample = [action_history[i] for i in indices]
                                done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])

                                # Build the updated Q-values for the sampled future states
                                # Use the target model for stability
                                future_rewards = target_network.predict(state_next_sample)
                                # Q value = reward + discount factor * expected future reward
                                updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)

                                # If final frame set the last value to -1
                                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                                # Create a mask so we only calculate loss on the updated Q-values
                                masks = tf.one_hot(action_sample, Flags.n_actions)

                                with tf.GradientTape() as tape:
                                    # Train the model on the states and updated Q-values
                                    q_values = q_network(state_sample, training=True)

                                    # Apply the masks to the Q-values to get the Q-value for action taken
                                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                                    # Calculate loss between new Q-value and old Q-value
                                    loss = loss_function(updated_q_values, q_action)

                                # Backpropagation
                                grads = tape.gradient(loss, q_network.trainable_variables)
                                optimizer.apply_gradients(zip(grads, q_network.trainable_variables))

                            if iter_count % update_target_network == 0:
                                # update the the target network with new weights
                                target_network.set_weights(q_network.get_weights())

                            # Limit the state and reward history
                            if len(rewards_history) > max_memory_length:
                                del rewards_history[:1]
                                del state_history[:1]
                                del state_next_history[:1]
                                del action_history[:1]
                                del done_history[:1]

                            image_count += 1

                        except tf.python.framework.errors_impl.OutOfRangeError:
                            done = True
                            print(iter_count)
                            break
                    
                    pbar.update(Flags.train_batch_size)
                    
                    total_episode_score += episode_score

                pat_feat, pat_diag = derm.update_patient(patients,n_words)
            
            print('')
            print('Finished Macro Episode ', macro_episode)
            print('The total episode score was ',total_episode_score)
            episode_reward_history.append(total_episode_score)

            ## Validation Phase ##
            state,pat_feat,pat_diag,patients_val = derm.reset(val_feat,val_labels, val_labels.shape[0], False,n_words)

            state = derm.state-np.concatenate([derm.context,derm.prob_context])

            management = np.array([])
            true_label = np.array([])

            actions_table = np.zeros([len(vocab),Flags.n_actions])

            for id_val in range(val_labels.shape[0]):

                done = False
                image_count = 1

                while not done:
                    try:
                        true_label = np.append(true_label, int(derm.gt))

                        diag = derm.gt

                        state_tensor = tf.convert_to_tensor(state)
                        state_tensor = tf.expand_dims(state_tensor, 0)
                        action_probs = q_network(state_tensor, training=False)
                        
                        # Take best action
                        action = tf.argmax(action_probs[0]).numpy()

                        management = np.append(management, action)

                        _, state, reward, done,_ = derm.step(pat_feat, pat_diag, Flags.n_actions, action,image_count)

                        state = derm.state-np.concatenate([derm.context,derm.prob_context])


                        episode_val_score += reward

                        actions_table[int(diag),action] +=1

                        image_count += 1

                    except tf.python.framework.errors_impl.OutOfRangeError:
                        done = True
                        break
                
                pat_feat, pat_diag = derm.update_patient(patients_val,n_words)
            
            if np.sum(actions_table[1,:]) > 0:

                mel_reward = np.dot(actions_table[1,:],[-5,-1,5])

                mel_qnt = np.sum(actions_table[1,:])
            else:
                mel_reward = 0
                mel_qnt = 1

            ben_reward = np.dot(actions_table[0,:],[5,2,-1])

            episode_val_score = mel_reward/np.sum(actions_table[1,:]) + ben_reward/np.sum(actions_table[0,:])

            print('The reward of the validation episode was ', episode_val_score)
            episode_val_reward_history.append(episode_val_score)

            if best_mel_reward <= mel_reward and macro_episode > 10:
                if best_ben_reward/np.sum(actions_table[0,:]) + best_mel_reward/mel_qnt <= ben_reward/np.sum(actions_table[0,:]) + mel_reward/ mel_qnt:
                    best_mel_reward = mel_reward
                    best_ben_reward = ben_reward
                    best_mel_actions_table = actions_table
                    q_network.save_weights('model/best_reward' + str(set),save_format='tf')
            

            print(actions_table)

            ##Return to train
            _,pat_feat, pat_diag,_ = derm.reset(train_feat,train_labels, train_labels.shape[0], True,n_words)

        print('The best reward scores:')
        print(best_mel_actions_table)
        print('The best reward was ', best_mel_reward+best_ben_reward)

        set += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--train_batch_size',
            type=int,
            default=1,
            help='Size of your batch.'
        )
    parser.add_argument(
        '--n_patients',
        type=int,
        default= 1,
        help='Number of patients until update.'
    )
    parser.add_argument(
        '--n_episodes',
        type=int,
        default= 120,
        help='Number of times to rotate patients'
    )
    parser.add_argument(
        '--n_actions',
        type=int,
        default= 3,
        help='Number of episodes to play'
    )
    Flags, unparsed = parser.parse_known_args()
    tf1.app.run(main=main)