import random
import numpy as np

class SnakeGame:

    def __init__(self, window_width, window_height, segment_size):
        self.snake = [((window_width // 2) - segment_size, window_height // 2),
                      ((window_width // 2) , window_height // 2),
                      ((window_width // 2) +segment_size, window_height // 2) ,
                      ((window_width // 2) +2*segment_size , window_height // 2)]
        self.snake_length = 1
        self.direction = 2
        self.food_position = ((window_width // 2) - 2*segment_size, window_height // 2)
        self.game_ended = 0
        self.segment_size = segment_size
        self.window_width = window_width
        self.window_height = window_height

    def reset(self):
        self.snake = [((self.window_width // 2) - self.segment_size, self.window_height // 2),
                      ((self.window_width // 2), self.window_height // 2),
                      ((self.window_width // 2) + self.segment_size, self.window_height // 2),
                      ((self.window_width // 2) + 2 * self.segment_size, self.window_height // 2)]
        self.snake_length = 1
        self.direction = 2
        self.food_position = ((self.window_width // 2) - 2 * self.segment_size, self.window_height // 2)
        self.game_ended = 0

    def random_food_position(self):
        x,y = self.food_position
        while (x,y) in self.snake[0:]:
            x = random.randint(0, (self.window_width - self.segment_size) // self.segment_size) * self.segment_size
            y = random.randint(0, (self.window_height - self.segment_size) // self.segment_size) * self.segment_size
        return (x, y)



    def get_state(self):

        head_x, head_y = self.snake[0][0], self.snake[0][1]

        left_tile = head_x - self.segment_size
        right_tile = head_x + self.segment_size
        upper_tile = head_y - self.segment_size
        bottom_tile = head_y + self.segment_size

        UP = 0
        BOTTOM = 1
        LEFT = 2
        RIGHT = 3

        def collides_with_tail(x, y):
            return ((x, y) in self.snake[1:])
        def wall_up():
            return upper_tile < 0
        def wall_down():
            return bottom_tile > self.window_height
        def wall_left():
            return left_tile < 0
        def wall_right():
            return right_tile > self.window_width


        vec = np.zeros(11)

        # snake direction
        vec[0] = self.direction == 0
        vec[1] = self.direction == 1
        vec[2] = self.direction == 2
        vec[3] = self.direction == 3



        # danger ahead
        vec[4] = (self.direction == UP and (wall_up() or collides_with_tail(head_x, upper_tile)) )or (
                self.direction == BOTTOM and (wall_down() or collides_with_tail(head_x, bottom_tile))) or (
                self.direction == LEFT and (wall_left() or collides_with_tail(left_tile, head_y )))or (
                self.direction == RIGHT and (wall_right() or collides_with_tail(right_tile, head_y)))

        # danger left
        vec[5] = (self.direction == UP and (wall_left() or collides_with_tail(left_tile, head_y )) )or (
                self.direction == BOTTOM and (wall_right() or collides_with_tail(right_tile, head_y))) or (
                self.direction == LEFT and (wall_down() or collides_with_tail(head_x, bottom_tile))) or (
                self.direction == RIGHT and (wall_up() or collides_with_tail(head_x, upper_tile)))

        # danger right
        vec[6] = (self.direction == UP and (wall_right() or collides_with_tail(right_tile, head_y)) ) or (
                self.direction == BOTTOM and (wall_left() or collides_with_tail(left_tile, head_y ))) or  (
                self.direction == LEFT and (wall_up() or collides_with_tail(head_x, upper_tile))) or (
                self.direction == RIGHT and (wall_down() or collides_with_tail(head_x, bottom_tile)))

        #food direction
        vec[7] = self.snake[0][1] < self.food_position[1] # food up
        vec[8] = self.snake[0][1] > self.food_position[1] # food down
        vec[9] = self.snake[0][0] < self.food_position[0] # food right
        vec[10] = self.snake[0][0] > self.food_position[0] # food left

        return vec

    def step(self, action):
        score = 0
        x, y = self.snake[0]

        if action == 0:  # ahead

            if self.direction == 0:

                y -= self.segment_size

            elif self.direction == 1:

                y += self.segment_size

            elif self.direction == 2:

                x -= self.segment_size

            elif self.direction == 3:

                x += self.segment_size

        elif action == 1:  # left

            if self.direction == 0:

                x -= self.segment_size
                self.direction = 2

            elif self.direction == 1:

                x += self.segment_size
                self.direction = 3

            elif self.direction == 2:

                y += self.segment_size
                self.direction = 1

            elif self.direction == 3:

                y -= self.segment_size
                self.direction = 0

        elif action == 2:  # right

            if self.direction == 0:

                x += self.segment_size
                self.direction = 3

            elif self.direction == 1:

                x -= self.segment_size
                self.direction = 2

            elif self.direction == 2:

                y -= self.segment_size
                self.direction = 0

            elif self.direction == 3:

                y += self.segment_size
                self.direction = 1

        self.snake.insert(0, (x, y))

        if self.snake[0] == self.food_position:
            self.snake_length += 1
            self.food_position = self.random_food_position()
            score = 10
        else:
            self.snake.pop()

        if (x < 0 or x >= self.window_width or y < 0 or\
            y >= self.window_height or self.snake[0] in self.snake[1:]):
            self.game_ended = 1
            score = -10


        return self.get_state(), score, self.game_ended
