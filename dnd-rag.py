import os
import requests
import json
import time
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Configuration
OLLAMA_API_ENDPOINT = 'http://localhost:11434/api/generate'
TURN_LIMIT = 10
PDF_FOLDER = 'pdf'  # Folder containing PDF files

# API call constants
TOP_P = 1
TOP_K = 40
TEMPERATURE = 0.8
PLAYER_MAX_TOKENS = 150  # Updated for 100-150 words
DM_MAX_TOKENS = 300      # Updated for 200-300 words
REPETITION_PENALTY = 1

# RAG Configuration
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = Chroma(embedding_function=embedding_model)

def load_pdfs(pdf_folder):
    """Load text from all PDFs in the specified folder."""
    documents = []
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(pdf_folder, file_name)
            with fitz.open(file_path) as pdf_doc:
                for page_num in range(pdf_doc.page_count):
                    page = pdf_doc[page_num]
                    documents.append(page.get_text("text"))
    return documents

def setup_vector_store(documents):
    """Create document embeddings and set up the vector store."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    splits = text_splitter.split_documents(documents)
    vector_store.add_documents(documents=splits)

def retrieve_documents(query):
    """Retrieve relevant documents based on a query."""
    retriever = vector_store.as_retriever()
    return retriever.retrieve(query)

def api_call(model_id, prompt, max_tokens):
    """Call the Ollama API to generate text based on the model ID and prompt."""
    data = {
        "model": model_id,
        "prompt": prompt,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
        "repetition_penalty": REPETITION_PENALTY
    }
    try:
        response = requests.post(OLLAMA_API_ENDPOINT, json=data, stream=True)
        response.raise_for_status()
        
        # Handle streaming response
        response_text = ""
        for line in response.iter_lines():
            if line:
                json_line = json.loads(line.decode('utf-8'))
                response_text += json_line.get('response', '')

        # Debugging: Print the concatenated response
        print(f"Concatenated Ollama API response: {response_text}")
        return {"output": {"choices": [{"text": response_text}]}}
    except requests.RequestException as e:
        print(f"API call error: {e}")
        return {"output": {"choices": [{"text": "Error in API call."}]}}

def generate_party():
    """Generate a party of LLM-controlled characters."""
    party_members = {}
    for i in range(4):
        prompt = f"Generate a D&D character with name, backstory, and items."
        response = api_call('phi3', prompt, PLAYER_MAX_TOKENS)
        character = response["output"]["choices"][0]["text"]
        party_members[f"Player {i+1}"] = character
    return party_members

def start_new_adventure(party_members):
    """Initialize a new adventure with a Dungeon Master (DM) introduction."""
    game_state = {
        "turn": 1,
        "story_progression": [],
        "turn_participation": {name: False for name in party_members},
        "current_turn": 0
    }
    dm_intro_prompt = "You are the Dungeon Master. Start the adventure and introduce the characters: " + ", ".join(party_members.keys())
    response = api_call('phi3', dm_intro_prompt, DM_MAX_TOKENS)
    dm_intro = response["output"]["choices"][0]["text"]
    game_state["story_progression"].append(dm_intro)
    print(f"DM: {dm_intro}\n")
    return game_state

def player_turn(player_name, player_info, game_state):
    """Handle the turn of a player character."""
    print(f"{player_name}'s turn:")
    player_prompt = f"{player_info}\nWhat do you do next?"
    response = api_call('phi3', player_prompt, PLAYER_MAX_TOKENS)
    player_action = response["output"]["choices"][0]["text"]
    print(f"{player_name}: {player_action}\n")
    game_state["story_progression"].append(f"{player_name}: {player_action}")
    game_state["turn_participation"][player_name] = True

def dm_turn(game_state):
    """Handle the turn of the Dungeon Master."""
    dm_prompt = "Summarize the actions and introduce the next challenge:\n" + "\n".join(game_state["story_progression"])
    response = api_call('phi3', dm_prompt, DM_MAX_TOKENS)
    dm_text = response["output"]["choices"][0].get('text', "Error: DM response does not contain 'text' key.")
    print(f"DM: {dm_text}\n")
    game_state["story_progression"].append(dm_text)
    game_state["turn_participation"] = {name: False for name in game_state["turn_participation"]}

def display_turn_info(game_state, party_members):
    """Display the current turn information and participation status."""
    print(f"\n--- Turn {game_state['turn']} ---")
    for player_name in party_members:
        status = "✓" if game_state["turn_participation"][player_name] else "✗"
        print(f"{player_name}: {status}")
    print()

def play_game(party_members):
    """Run the game loop, handling player and DM turns."""
    game_state = start_new_adventure(party_members)

    while game_state["turn"] <= TURN_LIMIT:
        display_turn_info(game_state, party_members)

        # Players' turns
        for player_name, player_info in party_members.items():
            if not game_state["turn_participation"][player_name]:
                player_turn(player_name, player_info, game_state)

        # DM's turn
        dm_turn(game_state)

        game_state["turn"] += 1
        game_state["current_turn"] += 1

        time.sleep(1)  # Adding a small delay for better readability

    print("Game over!")

def main():
    """Main menu for the D&D Simulation."""
    party_members = {}
    # Load and set up RAG
    documents = load_pdfs(PDF_FOLDER)
    setup_vector_store(documents)
    while True:
        print("Welcome to D&D LLM!")
        print("1. Generate Party")
        print("2. Start New Adventure")
        print("3. Quit")
        choice = input("Enter your choice: ")

        if choice == "1":
            party_members = generate_party()
        elif choice == "2":
            if party_members:
                play_game(party_members)
            else:
                print("You need to generate a party first!")
        elif choice == "3":
            break
        else:
            print("Invalid choice. Try again!")

if __name__ == "__main__":
    main()
