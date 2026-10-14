
import os
import sys
import torch
import matplotlib.pyplot as plt

from model import GPTLanguageModel
from data import load_data, get_batch


# Configuration
BATCH_SIZE = 32
BLOCK_SIZE = 128
MAX_ITERS = 3000
EVAL_INTERVAL = 300
LEARNING_RATE = 3e-4

N_EMBD = 128
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.1

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

torch.manual_seed(1337)
os.makedirs("outputs", exist_ok=True)

print("Training device:", DEVICE)

# Load the dataset and vocabulary
train_data, val_data, stoi, itos = load_data()

vocab_size = len(stoi)

# Create model
model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=N_EMBD,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    block_size=BLOCK_SIZE,
    dropout=DROPOUT
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


@torch.no_grad()
def estimate_loss():
    model.eval()
    results = {}

    for split, data in [
        ("train", train_data),
        ("val", val_data)
    ]:
        losses = []

        for _ in range(20):
            x, y = get_batch(
                data,
                batch_size=BATCH_SIZE,
                block_size=BLOCK_SIZE
            )

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            _, loss = model(x, y)
            losses.append(loss.item())

        results[split] = sum(losses) / len(losses)

    model.train()
    return results


train_losses = []
val_losses = []

model.train()

for step in range(MAX_ITERS):

    if step % EVAL_INTERVAL == 0:
        losses = estimate_loss()

        train_losses.append(losses["train"])
        val_losses.append(losses["val"])

        print(
            f"Step {step}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

    x, y = get_batch(
        train_data,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE
    )

    x = x.to(DEVICE)
    y = y.to(DEVICE)

    _, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0
    )

    optimizer.step()


# Save checkpoint
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "config": {
            "vocab_size": vocab_size,
            "n_embd": N_EMBD,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "block_size": BLOCK_SIZE,
            "dropout": DROPOUT
        }
    },
    "outputs/gpt_model.pt"
)

print("Model saved to outputs/gpt_model.pt")

# Plot training and validation losses
plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("Evaluation checkpoint")
plt.ylabel("Cross-entropy loss")
plt.title("NanoGPT Training Progress")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/loss_curve.png")
plt.show()
