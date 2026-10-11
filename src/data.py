
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "input.txt"


def load_text():
    return DATA_PATH.read_text(encoding="utf-8")


def build_vocabulary(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text, stoi):
    return [stoi[ch] for ch in text]


def decode(ids, itos):
    return "".join(itos[i] for i in ids)


def load_data():
    text = load_text()
    stoi, itos = build_vocabulary(text)

    data = torch.tensor(encode(text, stoi), dtype=torch.long)

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    return train_data, val_data, stoi, itos


def get_batch(data, batch_size=4, block_size=8):
    if len(data) <= block_size:
        raise ValueError("Dataset is too short for block_size.")

    starts = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size] for i in starts.tolist()
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in starts.tolist()
    ])

    return x, y


if __name__ == "__main__":
    train_data, val_data, stoi, itos = load_data()

    x, y = get_batch(train_data)

    print("Vocabulary size:", len(stoi))
    print("Training tokens:", len(train_data))
    print("Validation tokens:", len(val_data))
    print("Input shape:", x.shape)
    print("Target shape:", y.shape)
    print("Input:", decode(x[0].tolist(), itos))
    print("Target:", decode(y[0].tolist(), itos))