import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as transforms


def collate_fn(batch):
    waveforms, transcripts = zip(*batch)
    max_time = max(w.shape[-1] for w in waveforms)
    # Padding dos waveforms (supondo formato (channels, time))
    waveforms_padded = torch.stack(
        [torch.nn.functional.pad(w, (0, max_time - w.shape[-1])) for w in waveforms]
    )

    transcript_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.long)
    transcripts_padded = pad_sequence(transcripts, batch_first=True, padding_value=0)

    return waveforms_padded, transcripts_padded, transcript_lengths


def decode_predictions(logits, vocab):
    """Decodifica as predições do modelo em texto (decodificação greedy)."""
    idx_to_char = {v: k for k, v in vocab.items()}
    _, pred_indices = torch.max(logits, dim=-1)  # (batch, time)
    decoded_texts = []
    for pred in pred_indices:
        current_word = []
        prev_char = None
        for idx in pred:
            char = idx_to_char.get(idx.item(), "")
            if char == "<blank>":
                prev_char = None
                continue
            if char != prev_char:
                current_word.append(char)
            prev_char = char
        decoded_texts.append("".join(current_word).replace("<unk>", "").strip())
    return decoded_texts
