import torch
import torchaudio
from torch.utils.data import Dataset

class LibriSpeechDataset(Dataset):
    def __init__(self, root_dir, url="train-clean-100", transform=None, vocab=None, download=False):
        self.dataset = torchaudio.datasets.LIBRISPEECH(root=root_dir, url=url, download=download)
        self.transform = transform
        self.vocab = vocab

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, _, transcript, _, _, _ = self.dataset[idx]
        if self.transform:
            waveform = self.transform(waveform)
        transcript_indices = [self.vocab.get(c.lower(), 1) for c in transcript]
        return waveform, torch.tensor(transcript_indices, dtype=torch.long)

def build_vocab(dataset):
    chars = set()
    for i in range(len(dataset)):
        _, _, transcript, _, _, _ = dataset[i]
        chars.update(transcript.lower())
    return {'<blank>': 0, '<unk>': 1, **{c: i+2 for i, c in enumerate(sorted(chars))}}
