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
        waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = self.dataset[idx]
        if self.transform:
            waveform = self.transform(waveform)
        if self.vocab is not None:
            transcript_indices = [self.vocab.get(c.lower(), 1) for c in transcript]
            transcript_tensor = torch.tensor(transcript_indices, dtype=torch.long)
        else:
            transcript_tensor = transcript  # fallback
        return waveform, transcript_tensor

def build_vocab(dataset):
    chars = set()
    for i in range(len(dataset)):
        _, _, transcript, _, _, _ = dataset[i]
        chars.update(transcript.lower())
    # Definir índices: '<blank>'=0, '<unk>'=1, e os demais caracteres a partir de 2
    vocab = {'<blank>': 0, '<unk>': 1}
    for i, c in enumerate(sorted(chars)):
        vocab[c] = i + 2
    return vocab
