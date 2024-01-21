import pygame

class GameClient:

    def __init__(self, window_width, window_height, segment_size):
        self.window_width = window_width
        self.window_height = window_height
        self.segment_size = segment_size

        pygame.init()
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()

        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.WHITE = (255, 255, 255)
        self.YELLOW = (230, 230, 0)
        self.DARK_YELLOW = (200, 200, 0)


    def draw_segment(self, x, y):
        pygame.draw.rect(self.window, self.WHITE, [x, y, self.segment_size, self.segment_size])

    def draw_head(self, x, y):
        pygame.draw.rect(self.window, self.GREEN, [x, y, self.segment_size, self.segment_size])

    def draw_score(self, score):
        font = pygame.font.Font(None, 36)
        text = font.render('Score: ' + str(score), True, self.WHITE)
        self.window.blit(text, (self.window_width - 120, 10))
