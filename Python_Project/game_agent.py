from collections import deque
import tensorflow as tf
import numpy as np
class GameAgent:

    def __init__(self):

        input_shape = (14,)
        self.n_outputs = 3

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation="elu", input_shape=input_shape),
            tf.keras.layers.Dense(50, activation="elu"),
            tf.keras.layers.Dense(50, activation="elu"),
            tf.keras.layers.Dense(self.n_outputs)
        ])

        self.replay_buffer = deque(maxlen=100000)

        self.batch_size = 1000
        self.discount_factor = 0.91
        self.optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-2)
        self.loss_fn = tf.keras.losses.mean_squared_error

    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_outputs)
        else:
            Q_values = self.model.predict(state[np.newaxis], verbose=0)[0]
            return Q_values.argmax()

    def sample_experiences(self):
        indices = np.random.randint(len(self.replay_buffer), size=self.batch_size)
        states, actions, rewards, next_states, truncateds = zip(*[self.replay_buffer[index] for index in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(truncateds)

    def play_one_step(self, env, state, epsilon):
        action = self.epsilon_greedy_policy(state, epsilon)
        next_state, reward, truncated = env.step(action)
        self.replay_buffer.append((state, action, reward, next_state, truncated))
        return next_state, reward, truncated

    def training_step(self):
        experiences = self.sample_experiences()
        states, actions, rewards, next_states, truncateds = experiences
        next_Q_values = self.model.predict(next_states, verbose=0)
        max_next_Q_values = next_Q_values.max(axis=1)
        # print(max_next_Q_values)
        # print(max_next_Q_values.shape)
        runs = 1.0 - (truncateds)  # episode is not truncated
        target_Q_values = rewards + runs * self.discount_factor * max_next_Q_values
        target_Q_values = target_Q_values.reshape(-1, 1)
        mask = tf.one_hot(actions, self.n_outputs)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))





