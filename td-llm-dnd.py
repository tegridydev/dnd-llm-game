import streamlit as st
import os
import requests
import json
from typing import Dict, Tuple
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="TD-LLM-DND", page_icon="🐉", layout="wide")

OLLAMA_API_ENDPOINT = os.getenv('OLLAMA_API_ENDPOINT', 'http://localhost:11434/api/generate')
PDF_FOLDER = os.getenv('PDF_FOLDER', 'pdf')
CHROMA_DB_DIR = os.getenv('CHROMA_DB_DIR', './chroma_db')
TURN_LIMIT = int(os.getenv('TURN_LIMIT', 10))

for dir in [PDF_FOLDER, CHROMA_DB_DIR]:
    os.makedirs(dir, exist_ok=True)


def set_fantasy_theme():
    st.markdown("""
    <style>
        body { color: #e0e0e0; background-color: #1a1a2e; font-family: 'Cinzel', serif; }
        .stButton>button { color: #ffd700; background-color: #4a0e0e; border: 2px solid #ffd700; }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea { color: #e0e0e0; background-color: #2a2a4e; }
        .stHeader { color: #ffd700; text-shadow: 2px 2px 4px #000000; }
        .sidebar .sidebar-content { background-color: #16213e; }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)


@st.cache_resource
def initialize_rag(embedding_model_name):
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    return Chroma(embedding_function=embedding_model, persist_directory=CHROMA_DB_DIR)


def check_ollama_availability():
    try:
        response = requests.get(OLLAMA_API_ENDPOINT.replace('/api/generate', '/api/version'), timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def api_call(model: str, prompt: str, max_tokens: int) -> str:
    try:
        response = requests.post(OLLAMA_API_ENDPOINT,
                                 json={"model": model, "prompt": prompt, "max_tokens": max_tokens},
                                 headers={"Content-Type": "application/json"},
                                 timeout=30)
        response.raise_for_status()


        raw_text = response.text
        json_objects = []
        while raw_text:
            try:
                json_obj, index = json.JSONDecoder().raw_decode(raw_text)
                json_objects.append(json_obj)
                raw_text = raw_text[index:].strip()
            except json.JSONDecodeError:
                break

        if not json_objects:
            st.error("API call error: No valid JSON objects found in response.")
            return "Error: No valid JSON objects found in response."


        combined_response = " ".join([obj.get("response", "") for obj in json_objects])
        return combined_response

    except requests.RequestException as e:
        st.error(f"API call error: {str(e)}")
        return f"Error: Unable to generate content. Please check Ollama status."


def generate_character(model: str) -> str:
    return api_call(model,
                    "Generate a D&D character with name, race, class, backstory, and items. Be creative and diverse.",
                    150)


def generate_party(model: str) -> Dict[str, str]:
    return {f"Player {i + 1}": generate_character(model) for i in range(4)}


def start_new_adventure(model: str, party_members: Dict[str, str]) -> Tuple[Dict, str]:
    dm_intro = api_call(model,
                        f"You are the Dungeon Master. Start an exciting and unique D&D adventure. Introduce the characters: {', '.join(party_members.keys())}. Set the scene and present an initial challenge or mystery.",
                        300)
    return {
        "turn": 1,
        "story_progression": [dm_intro],
        "turn_participation": {name: False for name in party_members},
        "party_members": party_members
    }, dm_intro


def player_turn(model: str, player_name: str, player_info: str, game_state: Dict, vector_store) -> str:
    context = vector_store.similarity_search(player_info, k=3)
    player_prompt = f"{player_info}\nGame context: {' '.join(game_state['story_progression'][-3:])}\nRelevant lore: {' '.join([doc.page_content for doc in context])}\nWhat do you do next? (Respond in character)"
    return api_call(model, player_prompt, 150)


def dm_turn(model: str, game_state: Dict, vector_store) -> str:
    context = vector_store.similarity_search(" ".join(game_state['story_progression'][-5:]), k=3)
    dm_prompt = f"As the Dungeon Master, consider the recent events:\n{' '.join(game_state['story_progression'][-5:])}\nRelevant lore: {' '.join([doc.page_content for doc in context])}\nSummarize the actions, introduce the next challenge or plot development, and describe the scene. Be creative and engaging."
    return api_call(model, dm_prompt, 300)


def list_ollama_models():
    try:
        response = requests.get(OLLAMA_API_ENDPOINT.replace('/api/generate', '/api/tags'))
        if response.status_code == 200:
            return [model['name'] for model in response.json().get('models', [])]
        return []
    except requests.RequestException:
        return []


def manage_models():
    st.header("Manage Models")

    models = list_ollama_models()
    if not models:
        st.warning("No Ollama models found or Ollama is not running.")
        return

    st.subheader("Available Ollama Models")
    for model in models:
        st.write(f"- {model}")

    st.subheader("Select Models")
    dm_model = st.selectbox("Dungeon Master Model", models,
                            index=models.index(st.session_state.get('dm_model', models[0])))
    player_model = st.selectbox("AI Player Model", models,
                                index=models.index(st.session_state.get('player_model', models[0])))
    embedding_model = st.selectbox("RAG Embedding Model", ["sentence-transformers/all-mpnet-base-v2",
                                                           "sentence-transformers/all-MiniLM-L6-v2"])

    if st.button("Save Model Selections"):
        st.session_state.dm_model = dm_model
        st.session_state.player_model = player_model
        st.session_state.embedding_model = embedding_model
        st.session_state.vector_store = initialize_rag(embedding_model)
        st.success("Model selections saved!")


def display_status_sidebar():
    st.sidebar.header("System Status")
    ollama_status = "Running" if check_ollama_availability() else "Not Running"
    st.sidebar.write(f"Ollama Status: {ollama_status}")
    st.sidebar.write(f"Models Set: {'Yes' if 'dm_model' in st.session_state else 'No'}")
    st.sidebar.write(f"RAG Initialized: {'Yes' if 'vector_store' in st.session_state else 'No'}")


def main():
    set_fantasy_theme()
    st.title("🐉 TD-LLM-DND")

    display_status_sidebar()

    if not check_ollama_availability():
        st.error("Ollama is not available. Please make sure it's running and configured correctly.")
        st.info("To start Ollama, open a terminal and run the 'ollama serve' command.")
        if st.button("Retry Connection"):
            st.experimental_rerun()
        return

    if 'game_state' not in st.session_state:
        st.session_state.game_state = None
    if 'party' not in st.session_state:
        st.session_state.party = None

    page = st.sidebar.radio("Navigation", ["Play Game", "Manage Models"])

    if page == "Play Game":
        if 'dm_model' not in st.session_state or 'player_model' not in st.session_state or 'vector_store' not in st.session_state:
            st.warning("Please set up your models in the 'Manage Models' section before playing.")
            return

        st.sidebar.header("Game Controls")
        if st.sidebar.button("🧙‍♂️ Generate New Party"):
            with st.spinner("Summoning brave adventurers..."):
                st.session_state.party = generate_party(st.session_state.player_model)
            st.success("Your party has assembled!")

        if st.session_state.party and st.sidebar.button("🗺️ Start New Adventure"):
            with st.spinner("Preparing an epic quest..."):
                st.session_state.game_state, _ = start_new_adventure(st.session_state.dm_model, st.session_state.party)
            st.success("Your adventure begins!")

        if st.sidebar.button("🔄 Reset Game"):
            st.session_state.game_state = None
            st.session_state.party = None
            st.success("Game reset. Ready for a new adventure!")

        if st.session_state.party:
            st.header("🦸‍♀️ Party Information")
            for name, info in st.session_state.party.items():
                with st.expander(name):
                    st.write(info)

        if st.session_state.game_state:
            st.header("📜 Adventure Log")
            for event in st.session_state.game_state["story_progression"]:
                st.text_area("", event, height=100, disabled=True)

            if st.button("🎲 Play Next Turn"):
                with st.spinner("The dice are rolling..."):
                    for player_name, player_info in st.session_state.game_state["party_members"].items():
                        if not st.session_state.game_state["turn_participation"][player_name]:
                            action = player_turn(st.session_state.player_model, player_name, player_info,
                                                 st.session_state.game_state, st.session_state.vector_store)
                            st.session_state.game_state["story_progression"].append(f"{player_name}: {action}")
                            st.session_state.game_state["turn_participation"][player_name] = True
                            st.text_area(player_name, action, height=100, disabled=True)

                    dm_response = dm_turn(st.session_state.dm_model, st.session_state.game_state,
                                          st.session_state.vector_store)
                    st.session_state.game_state["story_progression"].append(dm_response)
                    st.text_area("Dungeon Master", dm_response, height=150, disabled=True)

                    st.session_state.game_state["turn"] += 1
                    st.session_state.game_state["turn_participation"] = {name: False for name in
                                                                         st.session_state.game_state[
                                                                             "turn_participation"]}

                    if st.session_state.game_state["turn"] > TURN_LIMIT:
                        st.warning(
                            f"The adventure has reached its end after {TURN_LIMIT} turns. Start a new game to continue playing!")

        if not st.session_state.game_state and not st.session_state.party:
            st.info("Generate a party and start a new adventure to begin your journey!")

    elif page == "Manage Models":
        manage_models()

    st.sidebar.info("""
    ## How to Play
    1. Generate New Party
    2. Start New Adventure
    3. Play Next Turn

    May your dice roll high!
    """)


if __name__ == "__main__":
    main()
