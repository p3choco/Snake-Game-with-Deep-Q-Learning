from pygame_client import GameClient
from snake_game import SnakeGame
from game_agent import GameAgent
import pygame
import matplotlib.pyplot as plt

window_width = 400
window_height = 400
segment_size = 20
snake_speed = 100
game_client = GameClient(window_width, window_height, segment_size)
game_agent = GameAgent()
game_agent.load_model("snake_model_560_7.0")
pygame.event.get() # on macos pygame needs to get event to properly work

game = SnakeGame(window_width, window_height, segment_size)
score_list = []

for episode in range(6000):
    game.reset()
    obs = game.get_state()
    for step in range(4000):

        obs, reward, truncated = game_agent.play_one_step(game, obs)
        game_client.window.fill(game_client.BLACK)
        game_client.draw_head(*game.snake[0])
        for segment in game.snake[1:]:
            game_client.draw_segment(*segment)
        pygame.draw.rect(game_client.window, game_client.RED, [game.food_position[0], game.food_position[1], segment_size, segment_size])

        game_client.draw_score(game.snake_length - 1)

        pygame.display.update()
        game_client.clock.tick(snake_speed)

        if truncated:
            break

