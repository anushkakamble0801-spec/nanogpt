
import torch
from model import GPTLanguageModel

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

checkpoint = torch.load(
    "outputs/gpt_model.pt",
    map_location=DEVICE,
    weights_only=False
)

config = checkpoint["config"]
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

model = GPTLanguageModel(**config).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Start with a newline character if available
start_id = stoi.get("\n", 0)
context = torch.tensor([[start_id]], device=DEVICE)

generated = model.generate(
    context,
    max_new_tokens=500,
    temperature=0.8
)

text = "".join(
    itos[int(token)] for token in generated[0].tolist()
)

print(text)
