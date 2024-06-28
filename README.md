# TD-LLM-DND

TD-LLM-DND is a Streamlit-based web application that creates a Dungeons & Dragons style adventure using locally hosted llm from ollama.

Players can generate characters, start new adventures, and progress through turns with an automated Dungeon Master, AI powered party members and player interactions (TO:DO).

![screen](/current/td-llm-dnd.png)

## Features

- **Generate D&D Characters**: Create unique characters with name, race, class, backstory, and items
- **Start New Adventure**: Begin a new adventure with the generated characters
- **Turn-Based Gameplay**: Progress through the adventure with player and Dungeon Master turns (TO:DO - add player input)
- **Manage Models**: Select/download and manage language models for the Dungeon Master and AI players

## Requirements

- Python 3.8+
- Streamlit
- Requests
- LangChain
- HuggingFace Transformers
- dotenv

## Installation

1. **Clone the repository**:
    ```
    git clone https://github.com/tegridydev/dnd-llm-game.git
    cd td-llm-dnd
    ```

2. **Create and activate a virtual environment**:
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    Create a `.env` file in the root directory with the following content:
    ```plaintext
    OLLAMA_API_ENDPOINT=http://localhost:11434/api/generate
    PDF_FOLDER=pdf
    CHROMA_DB_DIR=./chroma_db
    TURN_LIMIT=10
    ```

## Usage

1. **Start Ollama**:
    ```
    ollama serve
    ```

2. **Run the Streamlit app**:
    ```
    streamlit run app.py
    ```

3. **Access the app**:
    Open your browser and go to `http://localhost:8501`.

## How to Play

1. Generate a new party.
2. Start a new adventure.
3. Play the next turn.

May your dice roll high!

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request!


