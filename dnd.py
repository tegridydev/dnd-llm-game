import requests
import json
import time

# Configuration
API_KEY = "your-api-key-here"
APP_NAME = "DND LLM"
DM_MODEL_ID = "Qwen/Qwen1.5-7B-Chat"
PLAYER_MODEL_IDS = ["Qwen/Qwen1.5-7B-Chat"] * 4
API_ENDPOINT = 'https://api.together.xyz/inference'
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "User-Agent": APP_NAME
}
TURN_LIMIT = 10

# API call constants
TOP_P = 1
TOP_K = 40
TEMPERATURE = 0.1
MAX_TOKENS = 100
REPETITION_PENALTY = 1

def api_call(model_id, prompt, max_tokens=MAX_TOKENS, top_p=TOP_P, top_k=TOP_K, temperature=TEMPERATURE, repetition_penalty=REPETITION_PENALTY):
    """Call the API to generate text based on the model ID and prompt."""
    data = {
        "model": model_id,
        "prompt": prompt,
        "top_p": top_p,
        "top_k": top_k,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "repetition_penalty": repetition_penalty
    }
    try:
        response = requests.post(API_ENDPOINT, json=data, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"API call error: {e}")
        return {}

def generate_party():
    """Generate the party members with their backstories and items."""
    party = {
        "Human Player": {"backstory": "A brave adventurer", "items": ["Longsword", "Leather armor", "Healing potion"], "model_id": None},
        "Eilif Stonefist": {"backstory": "A former soldier seeking redemption", "items": ["Longsword", "Leather armor", "Healing potion"], "model_id": PLAYER_MODEL_IDS[0]},
        "Elara Moonwhisper": {"backstory": "A young wizard eager to prove themselves", "items": ["Spell component pouch", "Quarterstaff", "Dagger"], "model_id": PLAYER_MODEL_IDS[1]},
        "Arin the Bold": {"backstory": "A cunning rogue with a mysterious past", "items": ["Short sword", "Leather armor", "Thieves' tools"], "model_id": PLAYER_MODEL_IDS[2]},
        "Morgran Ironfist": {"backstory": "A devout cleric on a mission from their god", "items": ["Warhammer", "Chain mail", "Shield"], "model_id": PLAYER_MODEL_IDS[3]}
    }
    print("Party generated!")
    return party

def print_ascii_art(title):
    """Print ASCII art for the given title."""
    art = {
        "title": """
 ____  _                 _      __    __  _______ _     _             
|  _ \| |               (_)     \ \  / / |__   __| |   (_)            
| |_) | | ___   ___  ___ _  ___  \ \/ /     | |  | |__  _ _ __   __ _ 
|  _ <| |/ _ \ / __|/ __| |/ __|  \  /      | |  | '_ \| | '_ \ / _` |
| |_) | | (_) | (__| (__| | (__   /  \      | |  | | | | | | | | (_| |
|____/|_|\___/ \___|\___|_|\___| /_/\_\     |_|  |_| |_|_| |_|\__, |
                                                                 __/ |
                                                                |___/ 
        """,
        "turn": """
  _____                    _            
 |_   _|                  | |           
   | |  _ __  _ __  _   _ | |_  ___ ___ 
   | | | '_ \| '__|| | | || __|/ __/ _ \
  _| |_| | | | |   | |_| || |_| (_|  __/
 |_____/_| |_|_|    \__,_| \__|\___\___|
        """
    }
    print(art.get(title, ""))

def start_new_adventure(party_members):
    """Start a new adventure and initialize the game state."""
    game_state = {"turn": 0, "story_progression": [], "current_turn": 0}
    print_ascii_art("title")
    print("New adventure started!\n")
    
    # User inputs adventure info/lore
    adventure_info = input("Enter the info/lore for the adventure: ")
    game_state["story_progression"].append(adventure_info)

    # DM starts the adventure
    prompt = (
        f"Adventure Info: {adventure_info}\n"
        "DM: Welcome to our D&D adventure! You are all brave adventurers seeking fortune and glory in the land of Eldoria. "
        "What happens next?\n"
        "Your response should be a brief introduction of the adventure setting."
    )
    dm_response = api_call(DM_MODEL_ID, prompt)
    
    dm_text = dm_response.get('output', {}).get('choices', [{}])[0].get('text', "Error: DM response does not contain 'text' key.")
    print(f"DM: {dm_text}\n")
    game_state["story_progression"].append(dm_text)

    # DM introduces each player
    for player_name, player_info in party_members.items():
        if player_name != "Human Player":
            player_intro = (
                f"DM: Introducing {player_name}, {player_info['backstory']}. "
                f"They are equipped with {', '.join(player_info['items'])}.\n"
                "Your response should briefly describe the player's appearance, skills, and notable items."
            )
            print(player_intro)
            game_state["story_progression"].append(player_intro)
    
    return game_state

def player_turn(player_name, player_info, game_state):
    """Handle a player's turn."""
    if player_name == "Human Player":
        prompt = f"{game_state['story_progression'][-1]}\n{player_name}: "
        user_input = input(prompt)
        print(f"{player_name}: {user_input}\n")
        game_state["story_progression"].append(user_input)
    else:
        prompt = (
            f"{game_state['story_progression'][-1]}\n{player_name}: "
            "Describe your character's actions in response to the current situation."
        )
        player_response = api_call(player_info["model_id"], prompt)
        
        player_text = player_response.get('output', {}).get('choices', [{}])[0].get('text', f"Error: {player_name} response does not contain 'text' key.")
        print(f"{player_name}: {player_text}\n")
        game_state["story_progression"].append(player_text)

def dm_turn(game_state):
    """Handle the DM's turn."""
    prompt = (
        f"{game_state['story_progression'][-1]}\nDM: "
        "Summarize what happened in the previous turn and introduce new events or challenges for the players."
    )
    dm_response = api_call(DM_MODEL_ID, prompt)
    
    dm_text = dm_response.get('output', {}).get('choices', [{}])[0].get('text', "Error: DM response does not contain 'text' key.")
    print(f"DM: {dm_text}\n")
    game_state["story_progression"].append(dm_text)

def play_game(party_members):
    """Run the game loop, handling player and DM turns."""
    game_state = start_new_adventure(party_members)

    while game_state["turn"] < TURN_LIMIT:
        print_ascii_art("turn")
        print(f"\n--- Turn {game_state['turn'] + 1} ---\n")

        # Players' turns
        for player_name, player_info in party_members.items():
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
        print_ascii_art("title")
        print("Welcome to D&D Simulation!")
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
