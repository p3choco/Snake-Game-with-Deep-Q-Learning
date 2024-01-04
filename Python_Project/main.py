import pygame
import random
import sys
import tensorFlow as tf
import numpy as np

pygame.init()

window_width, window_height = 600, 400
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption('Snake Game')

BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (230, 230, 0)
DARK_YELLOW = (200, 200, 0)

segment_size = 20
snake_speed = 10

def random_food_position():
    x = random.randint(0, (window_width - segment_size) // segment_size) * segment_size
    y = random.randint(0, (window_height - segment_size) // segment_size) * segment_size
    return (x, y)

def draw_segment(x, y):
    pygame.draw.rect(window, WHITE, [x, y, segment_size, segment_size])

def draw_head(x, y):
    pygame.draw.rect(window, GREEN, [x, y, segment_size, segment_size])

def draw_score(score):
    font = pygame.font.Font(None, 36)
    text = font.render('Score: ' + str(score), True, WHITE)
    window.blit(text, (window_width - 120, 10))

snake = [(window_width // 2, window_height // 2)]
snake_length = 1
direction = None

food_position = random_food_position()

clock = pygame.time.Clock()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                direction = 'UP'
            elif event.key == pygame.K_DOWN:
                direction = 'DOWN'
            elif event.key == pygame.K_LEFT:
                direction = 'LEFT'
            elif event.key == pygame.K_RIGHT:
                direction = 'RIGHT'

    x, y = snake[0]
    if direction == 'UP':
        y -= segment_size
    elif direction == 'DOWN':
        y += segment_size
    elif direction == 'LEFT':
        x -= segment_size
    elif direction == 'RIGHT':
        x += segment_size
    snake.insert(0, (x, y))

    if snake[0] == food_position:
        snake_length += 1
        food_position = random_food_position()
    else:
        snake.pop()

    if (x < 0 or x >= window_width or y < 0 or y >= window_height or snake[0] in snake[1:]):
        break

    window.fill(BLACK)
    draw_head(*snake[0])
    for segment in snake[1:]:
        draw_segment(*segment)
    pygame.draw.rect(window, RED, [food_position[0], food_position[1], segment_size, segment_size])

    draw_score(snake_length - 1)

    pygame.display.update()
    clock.tick(snake_speed)


actions = ['UP', 'DOWN','LEFT', 'RIGHT']
apple_price = 500
every_move_threat = -10
given_things = snake, food_position[0], food_position[1]

input_shape = ??????
n_outputs = 4

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="elu", input_shape=input_shape),
    tf.keras.layers.Dense(32, activation="elu"),
    tf.keras.layers.Dense(n_outputs)
])

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        Q_values = model.predict(state[np.newaxis], verbose=0)[0]
        return Q_values.argmax()

from collections import deque

replay_buffer = deque(maxlen=2000)

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    return [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(6)
    ]

def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, truncated, info = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done, truncated))
    return next_state, reward, done, truncated, info

batch_size = 32
discount_factor = 0.95
optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-2)
loss_fn = tf.keras.losses.mean_squared_error

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones, truncateds = experiences
    next_Q_values = model.predict(next_states, verbose=0)
    max_next_Q_values = next_Q_values.max(axis=1)
    runs = 1.0 - (dones | truncateds)  # episode is not done or truncated
    target_Q_values = rewards + runs * discount_factor * max_next_Q_values
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

for episode in range(600):
    obs, info = env.reset()
    for step in range(200):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done, truncated, info = play_one_step(env, obs, epsilon)
        if done or truncated:
            break

    if episode > 50:
        training_step(batch_size)