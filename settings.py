import pygame

# appearance and arena settings
GRID_SIZE = 20

WIDTH = 28 * GRID_SIZE
HEIGHT = 28 * GRID_SIZE

TOP_BUFF = 2 * GRID_SIZE # buffer between the game arena and the top and bottom edge of the screen
LEFT_BUFF = 2 * GRID_SIZE # buffer between the game arena and the left edge of the screen

SCREEN_WIDTH = 46 * GRID_SIZE
SCREEN_HEIGHT = HEIGHT + 2*TOP_BUFF

WHITE = pygame.Color('#f1f2da')
RED = pygame.Color('#ff7777')
GREEN = pygame.Color('#ffce96')
BLACK = pygame.Color('#00303b')

# list of FPS to speed up or slow down game.
FPS = [2000, 1000, 500, 100, 50, 30, 10, 5]

# Model settings
HIDDEN_LAYER_SIZE = 128
LR = 0.001 # learning rate
GAMMA = 0.95
TAU = 0.01
EPSILON = 1
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.001


# set to true to track model progress and to plot results after training
tracking = False

# location to save model weights to
checkpoint_path = 'snake_model.pth'
