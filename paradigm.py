import os
import sys
import urllib.request
from gpt4all import GPT4All

# --- TUI IMPORTS ---
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll, Container
from textual.widgets import Input, Static
from textual import work

# --- RICH IMPORTS ---
from rich.markdown import Markdown 

# --- 1. MODEL & RESOURCE SETUP ---
model_filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
model_url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

if not os.path.exists(model_filename):
    print("Downloading model...")
    urllib.request.urlretrieve(model_url, model_filename)

try:
    # device='cpu' prevents DLL errors
    model = GPT4All(model_filename, model_path=".", allow_download=False, device='cpu')
except Exception as e:
    print(f"Error: {e}")
    sys.exit()

# Load Logo
try:
    with open("design.txt", "r", encoding="utf-8") as f:
        LOGO_ART = f.read().strip()
except FileNotFoundError:
    LOGO_ART = "PARADIGM AI"

SYSTEM_PROMPT = """<|system|>
You are Paradigm, a helpful AI. Answer concisely.
</s>
"""

# --- 2. THE "LIQUID" DESIGN (CSS) ---
CSS = """
Screen {
    layout: vertical;
    background: #111111;
}

#header_box {
    dock: top;
    height: auto;
    border: round magenta;
    color: magenta;
    text-align: center;
    padding: 0 1;
    margin-bottom: 1;
}

#chat_view {
    height: 1fr;
    border: none;
    scrollbar-size: 1 1;
    overflow-y: scroll;
}

Input {
    dock: bottom;
    border: round green;
    color: green;
    height: 3;
    margin-top: 1;
    background: #000000;
}

/* --- ALIGNMENT CONTAINERS --- */
/* These hold the bubbles and push them left or right */

.msg_row {
    width: 100%;
    height: auto;
    min-height: 1;
    padding-bottom: 1;
}

.user_row {
    align: right top; /* Forces user content to the RIGHT */
}

.ai_row {
    align: left top;  /* Forces AI content to the LEFT */
}

/* --- BUBBLE STYLING --- */

.bubble {
    width: auto;
    max-width: 80%;
    height: auto;
    padding: 0 2;
}

.user_bubble {
    border: round green;
    color: #ccffcc;
    background: #002200;
}

.ai_bubble {
    border: round cyan;
    color: #ccffff;
    background: #002222;
}
"""

# --- 3. CUSTOM WIDGETS ---

class Message(Static):
    """A single chat bubble widget."""
    pass

# --- 4. THE MAIN APPLICATION ---

class ParadigmApp(App):
    CSS = CSS
    
    chat_history = [] 

    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Static(LOGO_ART, id="header_box")
        
        with VerticalScroll(id="chat_view"):
            # Initial Welcome Message
            with Container(classes="msg_row ai_row"):
                yield Message(Markdown("System Online. Awaiting Input..."), classes="bubble ai_bubble")
        
        yield Input(placeholder="Type your message...", id="input_box")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Happens when you press Enter."""
        user_text = event.value.strip()
        if not user_text:
            return

        self.query_one(Input).value = ""
        
        # 1. Show User Message
        self.add_message(user_text, "user")
        
        # 2. Add to history
        self.chat_history.append({"role": "user", "content": user_text})
        
        # 3. Start AI Thinking
        self.generate_response_stream(user_text)

    def add_message(self, text, role):
        """Adds a new bubble inside an alignment container."""
        chat_view = self.query_one("#chat_view")
        
        # 1. Determine classes
        row_class = "user_row" if role == "user" else "ai_row"
        bubble_class = "user_bubble" if role == "user" else "ai_bubble"
        
        # 2. Create the container and bubble
        row_container = Container(classes=f"msg_row {row_class}")
        bubble = Message(Markdown(text), classes=f"bubble {bubble_class}")
        
        # 3. Mount PARENT first, THEN child (Fixes MountError)
        chat_view.mount(row_container) 
        row_container.mount(bubble)
        
        # 4. Scroll to bottom
        bubble.scroll_visible() 
        
        return bubble

    @work(exclusive=True, thread=True)
    def generate_response_stream(self, user_input):
        """Runs the AI in a separate thread."""
        
        full_prompt = SYSTEM_PROMPT
        for msg in self.chat_history[-4:]:
            tag = "<|user|>" if msg['role'] == 'user' else "<|assistant|>"
            full_prompt += f"{tag}\n{msg['content']}</s>\n"
        full_prompt += f"<|user|>\n{user_input}</s>\n<|assistant|>\n"

        # Create empty bubble
        ai_widget = self.call_from_thread(self.add_message, "...", "ai")
        
        full_response = ""
        
        # Stream response
        tokens = model.generate(full_prompt, max_tokens=300, streaming=True, temp=0.7)
        
        for token in tokens:
            full_response += token
            # Update text with cursor
            self.call_from_thread(ai_widget.update, Markdown(full_response + "█"))
        
        # Final update
        self.call_from_thread(ai_widget.update, Markdown(full_response))
        self.chat_history.append({"role": "ai", "content": full_response})

if __name__ == "__main__":
    app = ParadigmApp()
    app.run()