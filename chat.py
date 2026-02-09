import random
import json
import torch
import nltk
import sys
import os
from nltk.stem.porter import PorterStemmer
from model import NeuralNet

# --- KEY ADDITION: Function to find paths inside the EXE ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
# -----------------------------------------------------------

# --- Helper Functions ---
stemmer = PorterStemmer()

# Force NLTK to look in the internal data folder if needed, 
# or download it if it's missing on the user's machine.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = [0.0] * len(all_words)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# --- Load the Brain (Using the new resource_path function) ---
device = torch.device('cpu') # Force CPU for the EXE to avoid CUDA errors on non-NVIDIA PCs

# WRAP YOUR FILE NAMES WITH resource_path()
with open(resource_path('intents.json'), 'r') as f:
    intents = json.load(f)

FILE = resource_path("data.pth")
data = torch.load(FILE, map_location=device) # Load to CPU

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# --- The Chat Loop ---
bot_name = "Parad/gm"
print("Let's chat! (type 'quit' to exit)")

while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(torch.tensor(X, dtype=torch.float32).numpy()).unsqueeze(0).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Calculate probability to check confidence
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if intent['tag'] == tag:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")