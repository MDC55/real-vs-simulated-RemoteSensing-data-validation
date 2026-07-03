
from pathlib import Path
import pickle


def save_pickle(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved to: {path}")


def load_pickle(path):
    path = Path(path)
    with open(path, 'rb') as f:
        return pickle.load(f)
