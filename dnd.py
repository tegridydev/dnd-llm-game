import requests
import json
import time

# Configuration
API_KEY = "your-api-key"  # Replace 'your-api-key' with your actual Together.ai API key
APP_NAME = "D&D LLM"
DM_MODEL_ID_TOGETHER = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
PLAYER_MODEL_IDS_TOGETHER = ["NousResearch/Nous-Hermes-2-Mistral-7B-DPO"] * 4
DM_MODEL_ID_OLLAMA = "phi3"
PLAYER_MODEL_IDS_OLLAMA = ["phi3"] * 4
TOGETHER_API_ENDPOINT = 'https://api.together.xyz/inference'
OLLAMA_API_ENDPOINT = 'http://localhost:11434/api/generate'
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "User-Agent": APP_NAME
}
TURN_LIMIT = 10

# API call constants
TOP_P = 1
TOP_K = 40
TEMPERATURE = 0.8
PLAYER_MAX_TOKENS = 150  # Updated for 100-150 words
DM_MAX_TOKENS = 300      # Updated for 200-300 words
REPETITION_PENALTY = 1

USE_OLLAMA = True  # Set this to False to use Together API

def api_call(model_id, prompt, max_tokens):
    """Call the appropriate API to generate text based on the model ID and prompt."""
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
        if USE_OLLAMA:
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
        else:
            response = requests.post(TOGETHER_API_ENDPOINT, json=data, headers=HEADERS)
            response.raise_for_status()
            return response.json()
    except requests.RequestException as e:
        print(f"API call error: {e}")
        return {}
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return {"error": "JSON decode error"}

def generate_party():
    """Generate the party members with their backstories and items."""
    player_model_ids = PLAYER_MODEL_IDS_OLLAMA if USE_OLLAMA else PLAYER_MODEL_IDS_TOGETHER
    party = {
        "Eilif Stonefist": {"backstory": "A former soldier seeking redemption", "items": ["Longsword", "Leather armor", "Healing potion"], "model_id": player_model_ids[0]},
        "Elara Moonwhisper": {"backstory": "A young wizard eager to prove themselves", "items": ["Spell component pouch", "Quarterstaff", "Dagger"], "model_id": player_model_ids[1]},
        "Arin the Bold": {"backstory": "A cunning rogue with a mysterious past", "items": ["Short sword", "Leather armor", "Thieves' tools"], "model_id": player_model_ids[2]},
        "Morgran Ironfist": {"backstory": "A devout cleric on a mission from their god", "items": ["Warhammer", "Chain mail", "Shield"], "model_id": player_model_ids[3]}
    }
    print("Party generated!")
    return party

def print_ascii_art():
    """Print ASCII art for the title."""
    art = """
 ____  _    _  ____     _        _ __  __ 
|  _ \\| |  | |/ __ \\   | |      | |  \\/  |
| | | | |  | | |  | |  | |      | | \\  / |
| |_| | |__| | |__| |  | |____  | | |\\/| |
|____/ \\____/ \\____/   |______| |_|_|  |_|
    """
    print(art)

def start_new_adventure(party_members):
    """Start a new adventure and initialize the game state."""
    game_state = {"turn": 1, "story_progression": [], "current_turn": 0}
    
    print_ascii_art()
    print("New adventure started!\n")
    
    # Prompt the human player for their details
    human_player_name = input("Enter your character's name: ")
    human_player_backstory = input("Enter your character's backstory: ")
    human_player_items = input("Enter your character's items (comma separated): ").split(", ")
    
    party_members["Human Player"] = {
        "name": human_player_name,
        "backstory": human_player_backstory,
        "items": human_player_items,
        "model_id": None
    }
    
    # Initialize turn participation
    game_state["turn_participation"] = {name: False for name in party_members}
    
    # User inputs adventure info/lore
    adventure_info = input("Enter the info/lore for the adventure: ")
    game_state["story_progression"].append(adventure_info)

    # DM starts the adventure and introduces players
    dm_model_id = DM_MODEL_ID_OLLAMA if USE_OLLAMA else DM_MODEL_ID_TOGETHER
    prompt = (
        f"Adventure Info: {adventure_info}\n"
        "DM: Welcome to our D&D adventure! You are all brave adventurers seeking fortune and glory in the land of Eldoria. "
        "What happens next?\n"
        "Your response should be a detailed introduction of the adventure setting. Please include the following:\n"
        "- The current environment and atmosphere.\n"
        "- Any notable landmarks or features.\n"
        "- The immediate goal or situation for the players.\n"
        "Your response should be between 200 and 300 words."
    )
    dm_response = api_call(dm_model_id, prompt, DM_MAX_TOKENS)
    
    dm_text = dm_response.get('output', {}).get('choices', [{}])[0].get('text', "Error: DM response does not contain 'text' key.")
    print(f"DM: {dm_text}\n")
    game_state["story_progression"].append(dm_text)

    # DM introduces each player
    for player_name, player_info in party_members.items():
        player_intro = (
            f"DM: Introducing {player_info['name'] if player_name == 'Human Player' else player_name}, {player_info['backstory']}. "
            f"They are equipped with {', '.join(player_info['items'])}.\n"
            "Your response should include the following:\n"
            "- A brief description of the player's appearance.\n"
            "- Their notable skills and abilities.\n"
            "- Any special items or equipment they carry.\n"
            "Your response should be between 100 and 150 words."
        )
        if player_name == "Human Player":
            player_text = f"{player_info['name']} is a brave adventurer. {player_info['backstory']}. They are equipped with {', '.join(player_info['items'])}."
        else:
            player_response = api_call(player_info["model_id"], player_intro, PLAYER_MAX_TOKENS)
            player_text = player_response.get('output', {}).get('choices', [{}])[0].get('text', f"Error: {player_name} response does not contain 'text' key.")
        
        print(f"DM: {player_text}\n")
        game_state["story_progression"].append(player_text)
    
    return game_state

def player_turn(player_name, player_info, game_state):
    """Handle a player's turn."""
    if player_name == "Human Player":
        prompt = f"{game_state['story_progression'][-1]}\n{player_info['name']}: Describe your character's actions in response to the current situation. Your response should be between 100 and 150 words."
        user_input = input(prompt)
        print(f"{player_info['name']}: {user_input}\n")
        game_state["story_progression"].append(user_input)
    else:
        prompt = (
            f"{game_state['story_progression'][-1]}\n{player_name}: "
            "Describe your character's actions in response to the current situation. "
            "Your response should be between 100 and 150 words."
        )
        player_response = api_call(player_info["model_id"], prompt, PLAYER_MAX_TOKENS)
        
        player_text = player_response.get('output', {}).get('choices', [{}])[0].get('text', f"Error: {player_name} response does not contain 'text' key.")
        print(f"{player_name}: {player_text}\n")
        game_state["story_progression"].append(player_text)
    
    game_state["turn_participation"][player_name] = True

def dm_turn(game_state):
    """Handle the DM's turn."""
    dm_model_id = DM_MODEL_ID_OLLAMA if USE_OLLAMA else DM_MODEL_ID_TOGETHER
    prompt = (
        f"{game_state['story_progression'][-1]}\nDM: "
        "Summarize what happened in the previous turn and introduce new events or challenges for the players. "
        "Your response should be between 200 and 300 words."
    )
    dm_response = api_call(dm_model_id, prompt, DM_MAX_TOKENS)
    
    dm_text = dm_response.get('output', {}).get('choices', [{}])[0].get('text', "Error: DM response does not contain 'text' key.")
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
    while True:
        print_ascii_art()
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
