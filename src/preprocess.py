import os
import torch
import argparse
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import random_split
import torchaudio

from dataset import LibriSpeechDataset, build_vocab


def preprocess_data(raw_dir, processed_dir, datasets_dir, url="train-clean-100"):
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)

    processed_dataset_path = os.path.join(processed_dir, "full_dataset.pt")
    if not os.path.exists(processed_dataset_path):
        print("Iniciando pré-processamento...")
        # Carrega o dataset bruto (download já foi feito)
        raw_dataset = torchaudio.datasets.LIBRISPEECH(
            root=raw_dir, url=url, download=False
        )
        # Cria o vocabulário
        vocab = build_vocab(raw_dataset)
        torch.save(vocab, os.path.join(processed_dir, "vocab.pt"))
        print("Vocabulário criado!")
        # Define o transform (MelSpectrogram)
        transform = MelSpectrogram(sample_rate=16000, n_mels=128, normalized=True)
        dataset = LibriSpeechDataset(
            raw_dir, url=url, transform=transform, vocab=vocab, download=False
        )
        torch.save(dataset, processed_dataset_path)
        print("Dataset processado salvo!")
        # Divisão em splits
        total_len = len(dataset)
        train_size = int(0.7 * total_len)
        val_size = int(0.15 * total_len)
        test_size = total_len - train_size - val_size
        splits = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        torch.save(splits[0], os.path.join(datasets_dir, "train_set.pt"))
        torch.save(splits[1], os.path.join(datasets_dir, "val_set.pt"))
        torch.save(splits[2], os.path.join(datasets_dir, "test_set.pt"))
        print("Splits salvos!")
    else:
        print("Dataset já pré-processado. Pulando etapa.")


def main(args):
    preprocess_data(args.raw_dir, args.processed_dir, args.datasets_dir, url=args.url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pré-processamento do dataset LibriSpeech"
    )
    parser.add_argument(
        "--raw_dir", type=str, default="./data/raw", help="Diretório dos dados brutos"
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="./data/processed",
        help="Diretório para salvar os dados processados",
    )
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default="./datasets",
        help="Diretório para salvar os splits",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="train-clean-100",
        help="Tipo de dataset (ex.: train-clean-100)",
    )
    args = parser.parse_args()
    main(args)
