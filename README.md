
# dnd-llm-game

MVP of an idea using multiple LLM models to simulate and play D&D (Local LLM via ollama support + together.ai API support).

## Overview

This project is a proof of concept (MVP) that demonstrates how to use multiple language models (LLMs) to simulate and play a Dungeons & Dragons (D&D) game. The script involves a Dungeon Master (DM) and several player characters, some controlled by LLMs and one controlled by a human player.

## Features

- **Generate Party**: Generates a party of LLM-controlled characters.
- **Human Player Input**: Prompts the human player to enter their character details.
- **Adventure Setup**: The DM starts the adventure and introduces all characters.
- **Turn-Based Gameplay**: Each character takes turns describing their actions, followed by the DM summarizing the events and introducing new challenges.
- **Local LLM Support**: Option to use local LLMs via the Ollama API for LLM-controlled characters and the DM.

## Getting Started

### Prerequisites

- Python 3
- `requests` library (install via `pip install requests`)

### Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/tegridydev/dnd-llm-game.git
    cd dnd-llm-game
    ```

2. Install the required Python packages:
    ```sh
    pip install requests
    ```

3. Set your Together API key in the script or as an environment variable:
    ```sh
    export TOGETHER_API_KEY='your_api_key'
    ```

4. (Optional) Set up the Ollama API for local LLM support. Ensure the Ollama API is running and accessible at the specified endpoint (default: `http://localhost:11434/api/generate`).

### Running the Script

1. Run the script:
    ```sh
    python dnd.py
    ```

2. Follow the on-screen prompts to generate a party and start a new adventure.

### Using Local LLM (Ollama)

To use the local LLM via the Ollama API, set the `USE_OLLAMA` flag to `True` in the `dnd.py` script:
```python
USE_OLLAMA = True  # Set this to False to use Together API
```

Ensure the Ollama API is running locally and accessible. The default endpoint is `http://localhost:11434/api/generate`.

## How It Works

1. **Generate Party**: The script generates a party of LLM-controlled characters.
2. **Human Player Input**: The human player is prompted to enter their character's name, backstory, and items.
3. **Adventure Setup**: The DM introduces the adventure setting and all characters.
4. **Turn-Based Gameplay**: Each character takes turns to describe their actions, and the DM summarizes the events and introduces new challenges.

## Expansion Ideas

This project is open source and can be expanded in numerous ways:

1. **Character Development**: Add more detailed character creation options, including classes, races, and abilities.
2. **Inventory Management**: Implement inventory management for both LLM-controlled and human-controlled characters.
3. **Battle System**: Create a more complex battle system with dice rolls, hit points, and special abilities.
4. **Story Progression**: Develop a more sophisticated story progression system with branching narratives based on player choices.
5. **Multiplayer**: Allow multiple human players to join the game and interact with LLM-controlled characters.
6. **Graphics and UI**: Enhance the game with graphical elements and a user-friendly interface.
7. **Integration with D&D Rulesets**: Integrate the game with official D&D rulesets and modules.
8. **Save and Load**: Implement save and load functionality to allow players to continue their adventures at a later time.

## Contributing

Contributions are welcome! Please fork the repository and build some cool stuff.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- This project uses the Together API and the Ollama API for LLM interactions.
- Inspired by the creative world of Dungeons & Dragons.
