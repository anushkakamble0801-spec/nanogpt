
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import load_data, get_batch


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding(idx)
        loss = None

        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, C),
                targets.reshape(B * T)
            )

        return logits, loss


if __name__ == "__main__":
    train_data, val_data, stoi, itos = load_data()
    x, y = get_batch(train_data)

    model = BigramLanguageModel(len(stoi))
    logits, loss = model(x, y)

    print("Input shape:", x.shape)
    print("Logits shape:", logits.shape)
    print("Initial loss:", loss.item())