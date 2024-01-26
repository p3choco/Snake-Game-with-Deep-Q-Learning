from pygame_client import GameClient
from snake_game import SnakeGame
from game_agent import GameAgent
import pygame
import matplotlib.pyplot as plt
from statistics import mean

DISPLAY = True

window_width = 400
window_height = 400
segment_size = 20
snake_speed = 100

game_agent = GameAgent()
if DISPLAY:
    game_client = GameClient(window_width, window_height, segment_size)
    pygame.event.get() # on macos pygame needs to get event to properly work

game = SnakeGame(window_width, window_height, segment_size)
score_list = []
avg_score_list = []

best_avg = 0

for episode in range(6000):
    game.reset()
    obs = game.get_state()
    print("Episode: ", episode)
    for step in range(4000):

        epsilon = max((300 - episode)/300, 0.01)
        obs, reward, truncated = game_agent.play_one_step(game, obs, epsilon)
        if DISPLAY:
            game_client.window.fill(game_client.BLACK)
            game_client.draw_head(*game.snake[0])
            for segment in game.snake[1:]:
                game_client.draw_segment(*segment)
            pygame.draw.rect(game_client.window, game_client.RED, [game.food_position[0],
                                    game.food_position[1], segment_size, segment_size])

            game_client.draw_score(game.snake_length - 1)

            pygame.display.update()
            game_client.clock.tick(snake_speed)

        if truncated:
            break
    score_list.append(game.snake_length - 1)

    if episode > 50:
        game_agent.training_step()

    avg_score = mean(score_list[-10:])
    avg_score_list.append(avg_score)


    if episode > 80 and avg_score > best_avg:
        plt.plot(list(range(1,episode+2)),avg_score_list)
        plt.show()
        best_avg = avg_score
        game_agent.save_model(episode)