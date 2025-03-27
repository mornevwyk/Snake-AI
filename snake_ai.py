from brain import Brain
from enum import Enum
import math
import matplotlib.pyplot as plt
import numpy as np
import pygame
import random
import settings
import torch


# Set up screen settings and globals
WIDTH = settings.WIDTH
HEIGHT = settings.HEIGHT

GRID_SIZE = settings.GRID_SIZE

TOP_BUFF = settings.TOP_BUFF
LEFT_BUFF = settings.LEFT_BUFF

SCREEN_WIDTH = settings.SCREEN_WIDTH
SCREEN_HEIGHT = settings.SCREEN_HEIGHT

WHITE = settings.WHITE
RED = settings.RED
GREEN = settings.GREEN
BLACK = settings.BLACK

FPS = settings.FPS

pygame.font.init()
FONT_BIG = pygame.font.SysFont('Roboto', 42)
FONT = pygame.font.SysFont('Roboto', 20)

# variables to keep track of for plotting
high_score = []
run_score = []
iterations = []

class Direction(Enum):
    '''Enum for tracking and comparing directions'''
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

class SnakeAI:
    def __init__(self, use_ai=True):
        self.use_ai = use_ai
        pygame.init()
        self.snake = [
            pygame.Vector2((WIDTH + LEFT_BUFF)/2, HEIGHT/2), 
            pygame.Vector2((WIDTH + LEFT_BUFF)/2, (HEIGHT + TOP_BUFF)/2 + 10), 
            pygame.Vector2((WIDTH + LEFT_BUFF)/2, (HEIGHT + TOP_BUFF)/2 + 20)
            ]
        self.tail = self.snake[-1]
        self.action = [0,1,0]
        self.direction = Direction.UP
        self.food = None
        self.spawn_food()
        self.brain: Brain = Brain(15, settings.HIDDEN_LAYER_SIZE) if use_ai else None # Create a snake brain if program is to be run with AI.
        self.epsilon = self.brain.epsilon if use_ai else None 
        self.epsilon_decay = self.brain.epsilon_decay if use_ai else None 
        self.epsilon_min = self.brain.epsilon_min if use_ai else None 
        self.score = 0
        self.high_score = 0
        self.step_counter = 0
        self.fps_list = settings.FPS
        self.fps_idx = 1
        self.fps = 1000 if use_ai else 5
        self.iterations = 1
        self.done = False

    def get_state(self):
        '''
        Returns the current state. the states returned are:
            snake head x and y (ints)
            food x and y (ints)
            tail x and y (ints)
            angle to food (radians/pi -> float)
            currrent direction (one-hot)
            danger points (one-hot)
        '''
        positionX = self.snake[0].x
        positionY = self.snake[0].y
        
        tailX = self.snake[-1].x
        tailY = self.snake[-1].y

        foodX = self.food.x
        foodY = self.food.y

        snake_length = len(self.snake) / (WIDTH * HEIGHT / (GRID_SIZE ** 2))

        angle_to_food = math.atan2(foodY - positionY, foodX - positionX) / math.pi

        # check for danger around the snake i.e. edge of map and snake body
        danger_points = np.zeros(3, dtype=np.float32)  # Ensure float values

        if self.direction == Direction.UP:
            directions_to_check = [(-1, 0), #left
                                   (0, -1), #straight
                                   (1, 0) #right
                                   ]
        elif self.direction == Direction.RIGHT:
            directions_to_check = [(0, -1), #left
                                   (1, 0), #straight
                                   (0, 1) #right
                                   ] 
        elif self.direction == Direction.DOWN:
            directions_to_check = [(1, 0), #left
                                   (0, 1), #straight
                                   (-1, 0) #right
                                   ] 
        elif self.direction == Direction.LEFT:
            directions_to_check = [(0, 1), #left
                                   (-1, 0), #straight
                                   (0, -1) #right
                                   ] 

        for i, (dx, dy) in enumerate(directions_to_check):
            new_point = pygame.Vector2(positionX + dx * GRID_SIZE, positionY + dy * GRID_SIZE)
            danger_points[i] = float(self.check_danger(new_point))  # Convert boolean to float

        direction_encoding = [
            int(self.direction == Direction.UP),
            int(self.direction == Direction.RIGHT),
            int(self.direction == Direction.DOWN),
            int(self.direction == Direction.LEFT)
        ]

        state =  torch.tensor([
            (positionX - LEFT_BUFF)/WIDTH, 
            (positionY - TOP_BUFF)/HEIGHT, 
            (foodX - LEFT_BUFF)/WIDTH, 
            (foodY - TOP_BUFF)/HEIGHT, 
            (tailX - LEFT_BUFF)/WIDTH, 
            (tailY - TOP_BUFF)/HEIGHT, 
            angle_to_food,
            snake_length,
            *direction_encoding, 
            *danger_points
        ])
        
        return state

    def step(self):
        '''Takes a step through the game by generating an action according to current state, calculates rewards and updates the state. Returns the new state.'''

        state = self.get_state() # snapshot of state for neural net 
        
        # epsilon greedy function for next state determination
        if random.random() < self.epsilon:
            # choose random action
            next_action = random.choice([torch.tensor([1, 0, 0]),torch.tensor([0, 1, 0]),torch.tensor([0, 0, 1])])
            action = torch.argmax(next_action)  # Random action (Left, Forward, Right)
        else:
            # use model to select next action
            next_action = self.brain.predict(state) # is one-hot encoded (1,0,0) || (0,1,0) || (0,0,1)
            action = torch.argmax(next_action) # convert OH to idx for next step

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        if action == 1: # go straight
            self.direction = self.direction
        if action == 0: # turn left
            # lists directions in anti-clockwise so that the next direction can easily be determined depending on the current direction. Can index in to list, add 1 to index -> get new direction. Same is implimented for turning right.
            dirs = [Direction.RIGHT, Direction.UP, Direction.LEFT, Direction.DOWN]
            idx = (dirs.index(self.direction) + 1) % 4
            self.direction = dirs[idx]
        elif action == 2: # turn right
            dirs = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            idx = (dirs.index(self.direction) + 1) % 4
            self.direction = dirs[idx]
        
        #TODO: simplify direction choosing. Can be more easily done with a single list of direction e.g. clockwise, then +1 for right, -1 for left.

        old_distance = abs(self.snake[0].x - self.food.x) + abs(self.snake[0].y - self.food.y) # keeping track of old distance to determine if we made progress closer to the food or not.
        
        self.move_snake()
        
        new_distance = abs(self.snake[0].x - self.food.x) + abs(self.snake[0].y - self.food.y)
        
        reward = 0 # set reward to 0, fresh reward after every action

        self.done = self.collisions_game_over()
        
        if self.done:
            reward = -20 # penalty for ending game
        elif self.collision_food():
            reward = 50 # reward for eating food
            self.step_counter = 0
            self.add_to_score()
        elif new_distance < old_distance:
            reward = 5
        else:
            reward = -2

        reward += 0.1 # reward for staying alive

        next_state = self.get_state() # snapshot of state after action
        
        '''
        Store experience in memory. 
        This keeps track of the state that we started with at the begining (state), 
        then the action we took (anext_action), 
        the reward for performing that action (reward), 
        the state that resulted after thaking the next action (next_state) 
        and then whether the last action casused the game to end or not (done)
        '''
        self.brain.store_memory((state, next_action, reward, next_state, self.done)) # save tuple to memory
        self.step_counter += 1
        
        return (self.get_state(), reward, self.done)

    def reset(self):
        '''Resets the game afer a game over.'''
        print(self.score)
        self.snake = [
            pygame.Vector2((WIDTH + LEFT_BUFF)/2, (HEIGHT + TOP_BUFF)/2), 
            pygame.Vector2((WIDTH + LEFT_BUFF)/2, (HEIGHT + TOP_BUFF)/2 + 10), 
            pygame.Vector2((WIDTH + LEFT_BUFF)/2, (HEIGHT + TOP_BUFF)/2 + 20)
            ]
        self.direction = Direction.UP
        self.food = None
        self.action = [0,1,0]
        self.spawn_food()
        self.score = 0
        self.done = False
        self.step_counter = 0

    def move_snake(self):
        '''Updates the snakes internal location of all segments. Return the new postion of all snake segments'''
        new_pos = [segment for segment in self.snake]
        move = pygame.Vector2(0,0)
        if self.direction == Direction.UP:
            move = pygame.Vector2(0,-1)
        elif self.direction == Direction.RIGHT:
            move = pygame.Vector2(1,0)
        elif self.direction == Direction.LEFT:
            move = pygame.Vector2(-1,0)
        elif self.direction == Direction.DOWN:
            move = pygame.Vector2(0,1)
    
        for i, segment in enumerate(new_pos):
            if i == 0:
                new_pos[i] = self.snake[i] + move*GRID_SIZE
            else:
                new_pos[i] = self.snake[i-1]

        self.snake = new_pos

    def collisions_game_over(self):
        '''
        Checks if the game has reached an end state: out of bounds or collided with itself
        Returns boolean
        '''
        point = self.snake[0]
        if point.x < LEFT_BUFF or point.x >= (WIDTH + LEFT_BUFF) or point.y < TOP_BUFF or point.y >= (HEIGHT + TOP_BUFF):
            return True

        if (point.x, point.y) in [(s.x, s.y) for s in self.snake[1:]]:
            return True  # Collides with its own body
        
        return False
    
    def collision_food(self):
        '''Returns a boolean of whether the snake has collided with food or not.'''
        head = self.snake[0]
        if head == self.food:
            self.spawn_food()
            self.snake.append(pygame.Vector2(self.tail.x, self.tail.y))
            self.tail = self.snake[-1]
            return True
        self.tail = self.snake[-1]
        return False        

    def add_to_score(self):
        self.score += 1
        if self.score > self.high_score:
            self.high_score = self.score
    
    def check_danger(self, point):
        '''
        Checks the location (point) for any danger i.e. game over obstacles (out of bounds or snake segment)
        Return boolean. True = danger at location, False = no danger
        '''
        if point.x < 10 or point.x > (WIDTH + 10) or point.y < 10 or point.y > (HEIGHT + 10):
            return True

        if (point.x, point.y) in [(s.x, s.y) for s in self.snake[1:]]:
            return True  # Collides with its own body
        
        return False

    def spawn_food(self):
        '''Spawns food ina new empty location'''
        new_food = pygame.Vector2(random.randint(int(LEFT_BUFF/GRID_SIZE), int((WIDTH + LEFT_BUFF)/GRID_SIZE)-1)*GRID_SIZE, random.randint(int(TOP_BUFF/GRID_SIZE), int((HEIGHT + TOP_BUFF)/GRID_SIZE)-1)*GRID_SIZE)
        
        if (new_food.x, new_food.y) in [(s.x, s.y) for s in self.snake]:
            self.spawn_food()  # Collides with its own body
        else:
            self.food = new_food
            return

    def render(self, screen):
        '''
        Handles all the rendering of the snake object in the game.
        '''
        for i, segment in enumerate(self.snake):
            if i == 0:
                pygame.draw.rect(screen, BLACK, pygame.Rect(segment.x, segment.y, GRID_SIZE, GRID_SIZE))
                pygame.draw.rect(screen, WHITE, pygame.Rect(segment.x+1, segment.y+1, GRID_SIZE-2, GRID_SIZE-2))
            else:
                pygame.draw.rect(screen, BLACK, pygame.Rect(segment.x, segment.y, GRID_SIZE, GRID_SIZE))
                pygame.draw.rect(screen, GREEN, pygame.Rect(segment.x+1, segment.y+1, GRID_SIZE-2, GRID_SIZE-2))

        pygame.draw.rect(screen, BLACK, pygame.Rect(self.food.x, self.food.y, GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(screen, RED, pygame.Rect(self.food.x + 1, self.food.y + 1, GRID_SIZE - 2, GRID_SIZE - 2))

    def render_score(self, screen: pygame.Surface):
        '''Renders the score on the screen (number apples eaten)'''
        game_text = FONT_BIG.render('SN-AI-KE', False, WHITE)
        screen.blit(game_text, (WIDTH + LEFT_BUFF + 10, 4))
        
        score_text = FONT.render(f'Score: {self.score}', False, WHITE)
        screen.blit(score_text, (WIDTH + LEFT_BUFF + 15, 40))
        
        high_score_text = FONT.render(f'High Score: {self.high_score}', False, WHITE)
        screen.blit(high_score_text, (WIDTH + LEFT_BUFF + 15, 60))

        if self.use_ai:
            iterations_text = FONT.render(f'Iterations: {self.iterations}', False, WHITE)
            screen.blit(iterations_text, (WIDTH + LEFT_BUFF + 15, 80))

            fps_text = FONT.render(f'FPS: {self.fps}', False, WHITE)
            screen.blit(fps_text, (WIDTH + LEFT_BUFF + 15, SCREEN_HEIGHT - TOP_BUFF))

        if len(run_score) > 25:
            mean_score = np.mean(run_score[-25:])
            mean_text = FONT.render(f'Mean Score: {mean_score}', False, WHITE)
            screen.blit(mean_text, (WIDTH + LEFT_BUFF + 15, 100))

    def render_boundary(self, screen):
        '''Renders arena boundary'''
        pygame.draw.rect(screen, BLACK, pygame.Rect(0, 0, SCREEN_WIDTH, LEFT_BUFF))
        pygame.draw.rect(screen, BLACK, pygame.Rect(0, 0, LEFT_BUFF, SCREEN_HEIGHT))
        pygame.draw.rect(screen, BLACK, pygame.Rect(0, SCREEN_HEIGHT-TOP_BUFF, SCREEN_WIDTH, TOP_BUFF))
        pygame.draw.rect(screen, BLACK, pygame.Rect(LEFT_BUFF+WIDTH, 0, SCREEN_WIDTH - (LEFT_BUFF+WIDTH) , SCREEN_HEIGHT))
        # top line
        pygame.draw.line(screen, WHITE, 
                            (LEFT_BUFF-1, TOP_BUFF-1), 
                            (WIDTH + LEFT_BUFF+1, TOP_BUFF-1)
                            ) 
        # left line 
        pygame.draw.line(screen, WHITE, 
                            (LEFT_BUFF-1, TOP_BUFF-1), 
                            (LEFT_BUFF -1, HEIGHT + TOP_BUFF + 1)
                            )
        # bottom line
        pygame.draw.line(screen, WHITE, 
                            (LEFT_BUFF-1, HEIGHT + TOP_BUFF + 1), 
                            (WIDTH + LEFT_BUFF + 1, HEIGHT + TOP_BUFF + 1)
                            )
        pygame.draw.line(screen, WHITE, 
                            (WIDTH + LEFT_BUFF + 1, TOP_BUFF - 1), 
                            (WIDTH + LEFT_BUFF + 1, HEIGHT + TOP_BUFF + 1)
                            )
               
            
    def run(self):
        '''
        Runs the game loop.
        Use the up and down arrow keys to speed up or slow down the FPS.
        '''
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        clock = pygame.time.Clock()
        running = True

        while running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if self.use_ai and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.fps_idx -= 1
                    elif event.key == pygame.K_DOWN:
                        self.fps_idx += 1
                if not self.use_ai and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.direction = Direction.UP
                    elif event.key == pygame.K_DOWN:
                        self.direction = Direction.DOWN
                    elif event.key == pygame.K_LEFT:
                        self.direction = Direction.LEFT
                    elif event.key == pygame.K_RIGHT:
                        self.direction = Direction.RIGHT
                    
            screen.fill(BLACK)

            if self.use_ai:
                self.step()

                if self.step_counter % 5 == 0:  # Train every 5 steps
                    self.brain.train(64)

                if self.done:
                    self.iterations += 1
                    high_score.append(self.high_score)
                    run_score.append(self.score)
                    iterations.append(self.iterations)
                    self.reset()
                    self.brain.train(500)

                if self.step_counter >= (100 + 150 * self.score):
                    self.iterations += 1
                    high_score.append(self.high_score)
                    run_score.append(self.score)
                    iterations.append(self.iterations)
                    self.reset()
                
                self.fps = self.fps_list[self.fps_idx % len(self.fps_list)]
            else:
                self.move_snake()
                if self.collisions_game_over():
                    self.reset()
                if self.collision_food():
                    self.add_to_score()

            self.render(screen)
            self.render_boundary(screen)
            self.render_score(screen)

            pygame.display.flip()
            
            clock.tick(self.fps)

        pygame.quit()


if __name__ == '__main__':
    import sys
    use_ai = "--ai" in sys.argv
    snakeAI = SnakeAI(use_ai)
    snakeAI.run()

    snakeAI.brain.save_model()

    if settings.tracking:
        plt.plot(iterations, high_score, label= 'high score')
        plt.legend()
        plt.xlabel('Snake game runs')
        plt.ylabel('Points')
        plt.show()
        
        plt.hist(run_score, density=True)
        plt.show()