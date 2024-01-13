from pygame_client import GameClient
from snake_game import SnakeGame
from game_agent import GameAgent
import pygame

window_width = 400
window_height = 400
segment_size = 20
snake_speed = 100
game_client = GameClient(window_width, window_height, segment_size)
game_agent = GameAgent()
pygame.event.get() # on macos pygame needs to get event to properly work

for episode in range(6000):
    game = SnakeGame(window_width, window_height, segment_size)
    obs = game.get_state()
    print("Episode: ", episode)
    for step in range(4000):

        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, truncated = game_agent.play_one_step(game, obs, epsilon)
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

    if episode > 50:
        game_agent.training_step()


