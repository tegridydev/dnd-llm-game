import os, sys, logging
import torch; torch.classes.__path__ = []    # avoid Streamlit watcher errors
import streamlit as st
from requests.exceptions import ConnectionError
from ollama._types import ResponseError

# ensure project root
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from core.settings import settings
from core.utils import last_sentences
from services.game_runner import GameRunner
from services.ollama_client import ollama_client
from core.utils import build_index
from core.pdf_utils import load_all_pdf_texts

logger = logging.getLogger(__name__)
st.set_page_config(page_title="TD-LLM-DND", layout="wide")

def display_party(party):
    st.subheader("🧙‍♂️ Party Sheet")
    cols = st.columns(len(party))
    for col, (name, char) in zip(cols, party.items()):
        with col.expander(name, expanded=False):
            data = char.model_dump()
            st.write(f"**Race:** {data['race']}  ")
            st.write(f"**Class:** {data['class']}  ")
            st.write("**Items:**")
            for it in data["items"]:
                st.write(f"- {it}")
            st.write("**Backstory (snippet):**")
            st.write(data["backstory"][:200] + "...")

def display_log(story):
    st.subheader("📜 Adventure Log")
    for i, line in enumerate(story, start=1):
        who, text = line.split(":", 1)
        icon = "🧙‍♂️" if who.strip()=="DM" else "🎲"
        with st.expander(f"Turn {i} - {icon}"):
            st.markdown(f"**{who}:** {text.strip()}")

def main():
    st.sidebar.title("TD-LLM-DND Settings")
    st.sidebar.write(f"- **Ollama Host:** `{settings.ollama_host}`")
    st.sidebar.write(f"- **Model:** `{settings.ollama_model}`")
    st.sidebar.write(f"- **Turn Limit:** {settings.turn_limit}")
    st.sidebar.write(f"- **RAG:** {settings.enable_rag}")

    # RAG PDF upload
    up = st.sidebar.file_uploader("Upload PDFs for lore", accept_multiple_files=True, type="pdf")
    if up:
        for f in up:
            dst = settings.pdf_folder / f.name
            dst.write_bytes(f.getbuffer())
        build_index()
        st.sidebar.success("PDF index rebuilt!")

    # init runner
    if "runner" not in st.session_state:
        st.session_state.runner = GameRunner()
    runner: GameRunner = st.session_state.runner
    gs = runner.state

    st.title("🗡️ TD-LLM-DND Adventure")

    # Phase: start → new party
    if gs.phase == "start":
        if st.button("Generate Party"):
            try:
                runner.new_party()
            except Exception as e:
                st.error(e)
        return  # re-render

    # Show party once generated
    if runner.party:
        display_party(runner.party)

    # Phase: ready to start
    if gs.phase == "start" and runner.party:
        if st.button("🐉 Start Adventure"):
            try:
                runner.start_adventure()
            except Exception as e:
                st.error(e)
        return

    # Phase: intro text
    if gs.phase == "intro":
        st.markdown(f"**Intro:** {gs.intro_text}")
        if st.button("▶️ Continue"):
            runner.request_options()
        return

    # Phase: choice
    if gs.phase == "choice":
        opts = gs.current_options
        with st.form(key=f"form_{gs.turn}", clear_on_submit=False):
            choice = st.radio("What will you do?", opts, key=f"choice_{gs.turn}")
            submit = st.form_submit_button("Submit Choice")
        if submit:
            runner.process_player_choice(opts.index(choice))
            runner.run_dm_turn()
        return

    # Phase: DM response shown (and loop back to options)
    if gs.phase == "dm_response":
        last = gs.story[-1]
        who, txt = last.split(":",1)
        st.markdown(f"**{who.strip()}:** {txt.strip()}")
        if st.button("▶️ Next Turn"):
            runner.request_options()
        # fall through to log

    # Always show log at end
    display_log(gs.story)

if __name__ == "__main__":
    main()
