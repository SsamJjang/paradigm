import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import sys

# --- UI IMPORTS (Install with: pip install rich) ---
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live

console = Console()

# ==============================================================================
# 1. CONFIG (Must match your training exactly!)
# ==============================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 64
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2

# Load Vocabulary (Quickly read the training file to get char mappings)
try:
    with open('chat.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
except FileNotFoundError:
    console.print("[bold red]Error:[/bold red] Could not find 'chat.txt'. Needed for character mapping!")
    sys.exit()

# ==============================================================================
# 2. THE BRAIN (Exact copy of your model structure)
# ==============================================================================
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, None

    def generate_stream(self, idx, max_new_tokens):
        # Generator function that yields one char at a time
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next.item()

# ==============================================================================
# 3. INITIALIZATION
# ==============================================================================
console.print("[yellow]Loading Brain...[/yellow]")
model = BigramLanguageModel()
try:
    # Ensure map_location handles CPU/GPU mismatch
    model.load_state_dict(torch.load("gpt_model.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
except FileNotFoundError:
    console.print("[bold red]Error:[/bold red] 'gpt_model.pth' not found. Run training first!")
    sys.exit()

# ==============================================================================
# 4. THE UI LOOP
# ==============================================================================

def stream_response(generator):
    full_text = ""
    # Live update panel
    with Live(Panel("", title="[bold cyan]AI[/bold cyan]", border_style="cyan"), refresh_per_second=12) as live:
        for token_id in generator:
            char = decode([token_id])
            full_text += char
            
            # --- INTELLIGENT CUT-OFF ---
            # If the AI tries to start a new User line, stop immediately.
            if "User:" in full_text:
                full_text = full_text.split("User:")[0].strip()
                live.update(Panel(full_text, title="[bold cyan]AI[/bold cyan]", border_style="cyan"))
                return # Stop generating
            
            # Show the "cursor" block █
            live.update(Panel(full_text + "█", title="[bold cyan]AI[/bold cyan]", border_style="cyan"))
            time.sleep(0.02) # Typing speed

    # Print final clean version without cursor
    console.print(Panel(full_text.strip(), title="[bold cyan]AI[/bold cyan]", border_style="cyan"))

console.clear()
console.print(Panel.fit("[bold magenta]NEURAL NETWORK ONLINE[/bold magenta]", border_style="magenta"))

while True:
    user_input = Prompt.ask("\n[bold green]You[/bold green]")
    if user_input.lower() in ["quit", "exit"]:
        break

    # Format input exactly like training data
    context_str = f"User: {user_input}\nBot:" 
    context = torch.tensor([encode(context_str)], dtype=torch.long, device=device)

    stream_response(model.generate_stream(context, max_new_tokens=200))