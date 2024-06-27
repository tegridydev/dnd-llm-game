import os
import requests
import json
import time
import fitz  # PyMuPDF
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OLLAMA_API_ENDPOINT = os.getenv('OLLAMA_API_ENDPOINT', 'http://localhost:11434/api/generate')
TURN_LIMIT = int(os.getenv('TURN_LIMIT', 10))
PDF_FOLDER = os.getenv('PDF_FOLDER', 'pdf')
MODEL_ID = os.getenv('MODEL_ID', 'phi3')

# API call constants
TOP_P = float(os.getenv('TOP_P', 1))
TOP_K = int(os.getenv('TOP_K', 40))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.8))
PLAYER_MAX_TOKENS = int(os.getenv('PLAYER_MAX_TOKENS', 150))
DM_MAX_TOKENS = int(os.getenv('DM_MAX_TOKENS', 300))
REPETITION_PENALTY = float(os.getenv('REPETITION_PENALTY', 1))

# RAG Configuration
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = Chroma(embedding_function=embedding_model, persist_directory="./chroma_db")

class APIError(Exception):
    """Custom exception for API-related errors."""
    pass

def load_pdfs(pdf_folder: str) -> List[str]:
    """Load text from all PDFs in the specified folder."""
    documents = []
    with ThreadPoolExecutor() as executor:
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        futures = [executor.submit(extract_text_from_pdf, os.path.join(pdf_folder, pdf_file)) for pdf_file in pdf_files]
        for future in futures:
            documents.extend(future.result())
    return documents

def extract_text_from_pdf(file_path: str) -> List[str]:
    """Extract text from a single PDF file."""
    with fitz.open(file_path) as pdf_doc:
        return [page.get_text("text") for page in pdf_doc]

def setup_vector_store(documents: List[str]):
    """Create document embeddings and set up the vector store."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_text("\n".join(documents))
    vector_store.add_texts(texts=splits)
    vector_store.persist()

def retrieve_documents(query: str) -> List[str]:
    """Retrieve relevant documents based on a query."""
    results = vector_store.similarity_search(query, k=3)
    return [doc.page_content for doc in results]

def api_call(model_id: str, prompt: str, max_tokens: int) -> Dict:
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
        response = requests.post(OLLAMA_API_ENDPOINT, json=data, stream=True, timeout=30)
        response.raise_for_status()
        
        response_text = "".join(json.loads(line)["response"] for line in response.iter_lines() if line)
        return {"output": {"choices": [{"text": response_text}]}}
    except requests.RequestException as e:
        raise APIError(f"API call error: {str(e)}")

def generate_party() -> Dict[str, str]:
    """Generate a party of LLM-controlled characters."""
    party_members = {}
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_character, i) for i in range(4)]
        for i, future in enumerate(futures):
            party_members[f"Player {i+1}"] = future.result()
    return party_members

def generate_character(index: int) -> str:
    """Generate a single character."""
    prompt = f"Generate a D&D character with name, race, class, backstory, and items. Be creative and diverse."
    response = api_call(MODEL_ID, prompt, PLAYER_MAX_TOKENS)
    return response["output"]["choices"][0]["text"]

def start_new_adventure(party_members: Dict[str, str]) -> Tuple[Dict, str]:
    """Initialize a new adventure with a Dungeon Master (DM) introduction."""
    game_state = {
        "turn": 1,
        "story_progression": [],
        "turn_participation": {name: False for name in party_members},
        "current_turn": 0,
        "party_members": party_members
    }
    dm_intro_prompt = f"You are the Dungeon Master. Start an exciting and unique D&D adventure. Introduce the characters: {', '.join(party_members.keys())}. Set the scene and present an initial challenge or mystery."
    response = api_call(MODEL_ID, dm_intro_prompt, DM_MAX_TOKENS)
    dm_intro = response["output"]["choices"][0]["text"]
    game_state["story_progression"].append(dm_intro)
    return game_state, dm_intro

def player_turn(player_name: str, player_info: str, game_state: Dict) -> str:
    """Handle the turn of a player character."""
    context = retrieve_documents(player_info)
    player_prompt = f"{player_info}\nGame context: {' '.join(game_state['story_progression'][-3:])}\nRelevant lore: {' '.join(context)}\nWhat do you do next? (Respond in character)"
    response = api_call(MODEL_ID, player_prompt, PLAYER_MAX_TOKENS)
    player_action = response["output"]["choices"][0]["text"]
    game_state["story_progression"].append(f"{player_name}: {player_action}")
    game_state["turn_participation"][player_name] = True
    return player_action

def dm_turn(game_state: Dict) -> str:
    """Handle the turn of the Dungeon Master."""
    context = retrieve_documents(" ".join(game_state["story_progression"][-5:]))
    dm_prompt = f"As the Dungeon Master, consider the recent events:\n{' '.join(game_state['story_progression'][-5:])}\nRelevant lore: {' '.join(context)}\nSummarize the actions, introduce the next challenge or plot development, and describe the scene. Be creative and engaging."
    response = api_call(MODEL_ID, dm_prompt, DM_MAX_TOKENS)
    dm_text = response["output"]["choices"][0]["text"]
    game_state["story_progression"].append(dm_text)
    game_state["turn_participation"] = {name: False for name in game_state["turn_participation"]}
    return dm_text

# Gradio interface functions
def generate_party_interface() -> str:
    """Generate a party and format it for display in the Gradio interface."""
    try:
        party = generate_party()
        return "\n\n".join([f"{name}:\n{info}" for name, info in party.items()])
    except APIError as e:
        return f"Error generating party: {str(e)}"

def start_adventure_interface(party_text: str) -> Tuple[str, str]:
    """Start a new adventure based on the provided party information."""
    try:
        party_members = {}
        for line in party_text.split('\n\n'):
            if line:
                name, info = line.split(':\n', 1)
                party_members[name] = info
        
        game_state, dm_intro = start_new_adventure(party_members)
        return dm_intro, json.dumps(game_state)
    except APIError as e:
        return f"Error starting adventure: {str(e)}", None

def play_turn_interface(game_state: str) -> Tuple[str, str]:
    """Play a single turn of the game."""
    if not game_state:
        return "Please start a new adventure first.", None
    
    try:
        game_state = json.loads(game_state)
        
        # Players' turns
        player_actions = []
        for player_name, player_info in game_state["party_members"].items():
            if not game_state["turn_participation"][player_name]:
                action = player_turn(player_name, player_info, game_state)
                player_actions.append(f"{player_name}: {action}")
        
        # DM's turn
        dm_response = dm_turn(game_state)
        
        game_state["turn"] += 1
        game_state["current_turn"] += 1
        
        turn_summary = "\n\n".join(player_actions + [f"DM: {dm_response}"])
        return turn_summary, json.dumps(game_state)
    except APIError as e:
        return f"Error playing turn: {str(e)}", json.dumps(game_state)

# Gradio interface
with gr.Blocks(title="D&D RAG-LLM Adventure") as app:
    gr.Markdown("# D&D RAG-LLM Adventure Game")
    
    with gr.Tab("Generate Party"):
        generate_btn = gr.Button("Generate New Party")
        party_output = gr.Textbox(label="Generated Party", lines=10)
        generate_btn.click(generate_party_interface, outputs=party_output)
    
    with gr.Tab("Play Game"):
        party_input = gr.Textbox(label="Party Information", lines=10)
        start_btn = gr.Button("Start New Adventure")
        game_output = gr.Textbox(label="Game Progress", lines=15)
        game_state = gr.State()
        
        start_btn.click(start_adventure_interface, inputs=[party_input], outputs=[game_output, game_state])
        
        play_turn_btn = gr.Button("Play Next Turn")
        play_turn_btn.click(play_turn_interface, inputs=[game_state], outputs=[game_output, game_state])

    gr.Markdown("""
    ## How to Play
    1. Go to the "Generate Party" tab and click "Generate New Party" to create your adventurers.
    2. Copy the generated party information.
    3. Switch to the "Play Game" tab and paste the party information into the "Party Information" box.
    4. Click "Start New Adventure" to begin your journey.
    5. Use the "Play Next Turn" button to progress through the adventure.
    
    Enjoy your unique D&D experience powered by AI!
    """)

# Initialize RAG system
print("Initializing RAG system...")
documents = load_pdfs(PDF_FOLDER)
setup_vector_store(documents)
print("RAG system initialized.")

# Launch the app
if __name__ == "__main__":
    app.launch()
