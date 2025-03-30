# Snake AI - Deep Q Learning

This project is an AI-powered Snake game where an agent learns to play using **Deep Q-Learning (DQN)**. The AI trains by optimizing its actions to maximize score while avoiding collisions.

## Features
- **Deep Q-Network (DQN)**: Neural network for decision-making.
- **Experience Replay**: Stores game experiences for better learning.
- **Target Network**: Stabilizes training with a secondary network.
- **Epsilon-Greedy Strategy**: Balances exploration vs. exploitation.
- **Pygame Visualization**: Watch the AI play in real-time.
- **Model Saving & Loading**: Continue training from saved checkpoints.
- **Manual Mode**: Play the game yourself and try to beat the score of your AI!

<img src=Screenshots/image.png width='520px'>

## Installation
### 1. Clone the repository
```sh
git clone https://github.com/yourusername/snake-ai.git
cd snake-ai
```
### 2. Install dependencies
```sh
pip install -r requirements.txt
```
### 3. Run the AI
To train and watch the AI play:
```sh
python snake_ai.py --ai
```
Control the frame rate using the up and down arrow keys.

To play the game manually:
```sh
python snake_ai.py
```
Control the snake with the arrow keys.

## Project Structure
```
├── brain.py        # Deep Q-Learning model
├── settings.py     # Game & training settings
├── snake_ai.py     # Main game loop & AI control
├── requirements.txt
└── README.md       # Project documentation
```

## Customization
Modify `settings.py` to adjust:
- Learning rate (`LR`)
- Hidden layer size
- Discount factor (`GAMMA`)
- Epsilon greedy settings (`EPSILON`)
- Soft update factor (`TAU`)
- Screen dimensions
- FPS settings
- Model checkpoint path
- Game colours

## Training Process
The AI learns through **reinforcement learning** by playing games and improving based on rewards:
- +50: Eating food
- -20: Colliding with the wall or itself
- Small rewards for moving toward food, penalties for moving away
- Small reward for staying alive

## Future Improvements
- Compare player scores to AI scores

## Contributing
Pull requests are welcome! Please open an issue for major changes.

## License
MIT License - Feel free to use and modify!

