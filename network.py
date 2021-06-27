import os
import random
import numpy as np
from collections import deque
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import load_model


class DQNetwork:
    def __init__(self, gamma: float, n_actions: int, epsilon: float,
                 batch_size: int, input_dims: int, lr: float = 0.00025,
                 epsilon_delta: float = 1e-5, epsilon_final: float = 0.01,
                 memory_size: int = 1000000, filename: str = "model.h5",
                 train: bool = True):
        """
        @param gamma: gamma parameter in Q-learning
        @param n_actions: number of possible actions
        @param epsilon: probability of random action.
        @param batch_size: size of batch.
        @param input_dims: model input dimension
        @param lr: learning rate
        @param epsilon_delta: value by which the epsilon is decreased
        @param epsilon_final: minimum epsilon value
        @param memory_size: maximum transitions stored
        @param filename: model filename
        @param train: True if new model has to be created,
                      otherwise load model from file
        """
        self.action_space = range(n_actions)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_delta = epsilon_delta
        self.epsilon_final = epsilon_final
        self.batch_size = batch_size
        self.model_file = filename
        self.memory = deque(maxlen=memory_size)
        self.model = (self.create_model(lr, n_actions, input_dims)
                      if train else self.load_model())

    @staticmethod
    def create_model(lr, n_actions, input_dims):
        """
        Method to define Keras model.
        @param n_actions: number of possible actions
        @param input_dims: model input dimension
        @return: model
        """
        model = Sequential()
        model.add(Dense(256, activation="relu", input_shape=(input_dims,)))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(n_actions))
        model.compile(loss="mse",
                      optimizer=Adam(lr=lr),
                      metrics=["accuracy"])

        return model

    def store_transition(self, transition: tuple):
        """
        Method to save current transition.
        @param transition: current transition
        """
        self.memory.append(transition)

    def choose_action(self, observation: list) -> int:
        """
        Method to predict action based on epsilon and observation.
        @observation: game state
        @return: action (direction)
        """
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array(observation)
            state = state[np.newaxis, :]
            actions = self.model.predict(state)
            action = np.argmax(actions)

        return action

    def train(self):
        """
        Method to train the model.
        """
        if len(self.memory) < 1000:
            return

        batch = np.array(random.sample(self.memory, self.batch_size))
        states, actions, rewards, next_states, dones = np.hsplit(batch, 5)
        states = np.array(list(map(lambda x: np.array(x[0]), states)))
        actions = np.squeeze(actions, axis=1).astype(int)
        rewards = np.squeeze(rewards, axis=1)
        next_states = np.array(list(map(lambda x: np.array(x[0]),
                                        next_states)))
        dones = np.squeeze(dones, axis=1)

        q_eval = self.model.predict(states)
        q_next = self.model.predict(next_states)

        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = (rewards + self.gamma *
                                          np.max(q_next, axis=1) * dones)

        self.model.train_on_batch(states, q_target)
        self.epsilon = (self.epsilon - self.epsilon_delta
                        if self.epsilon > self.epsilon_final
                        else self.epsilon_final)

    def save_model(self):
        """
        Method to save the model.
        """
        self.model.save(self.model_file)

    def load_model(self):
        """
        Method to load the model.
        @return: model
        """
        return load_model(self.model_file)
