import os
import torch
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn

from utils import collate_fn

def train(model, train_loader, val_loader, device, model_dir, lr=3e-4, epochs=50, patience=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=3, factor=0.5
    )
    criterion = nn.CTCLoss(blank=0)

    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for waveforms, transcripts, lengths in train_loader:
            waveforms = waveforms.to(device)
            transcripts = transcripts.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            logits = model(waveforms)
            input_lengths = torch.full(
                (waveforms.size(0),), logits.size(1), dtype=torch.long, device=device
            )
            loss = criterion(
                logits.permute(1, 0, 2), transcripts, input_lengths, lengths
            )
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for waveforms, transcripts, lengths in val_loader:
                waveforms = waveforms.to(device)
                transcripts = transcripts.to(device)
                lengths = lengths.to(device)
                logits = model(waveforms)
                input_lengths = torch.full(
                    (waveforms.size(0),),
                    logits.size(1),
                    dtype=torch.long,
                    device=device,
                )
                val_loss = criterion(
                    logits.permute(1, 0, 2), transcripts, input_lengths, lengths
                )
                total_val_loss += val_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_without_improvement = 0
            os.makedirs(model_dir, exist_ok=True)
            best_model_path = os.path.join(model_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"Novo melhor modelo salvo na epoch {epoch + 1}")
        else:
            epochs_without_improvement += 1
            print(f"Sem melhoria por {epochs_without_improvement} epoch(s)")
            if epochs_without_improvement >= patience:
                print("Early stopping acionado.")
                break

    final_model_path = os.path.join(model_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Modelo final salvo em {final_model_path}")


def main(args):
    datasets_dir = args.datasets_dir
    processed_dir = args.processed_dir
    model_dir = os.path.join(args.models_dir, args.model_name)

    # Carrega os splits e o vocabulário
    train_set = torch.load(os.path.join(datasets_dir, "train_set.pt"))
    val_set = torch.load(os.path.join(datasets_dir, "val_set.pt"))
    vocab = torch.load(os.path.join(processed_dir, "vocab.pt"))

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate_fn)

    # Importa as variantes de modelo
    from models import TransformerASR_DA, ConformerASRBase, ConformerASRPE, ConformerASRDA

    # Mapeia o nome da variante à classe correspondente
    model_variants = {
        "TransformerASR_DA": TransformerASR_DA,
        "ConformerASRBase": ConformerASRBase,
        "ConformerASRPE": ConformerASRPE,
        "ConformerASRDA": ConformerASRDA,
    }

    if args.model_variant not in model_variants:
        raise ValueError(
            f"Modelo variante '{args.model_variant}' não suportado. Escolha entre: {list(model_variants.keys())}"
        )

    ModelClass = model_variants[args.model_variant]
    model = ModelClass(
        input_size=args.input_size,
        vocab_size=len(vocab),
        dim_model=args.dim_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )

    device = args.device if torch.cuda.is_available() else "cpu"
    train(
        model,
        train_loader,
        val_loader,
        device,
        model_dir,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Treinamento do modelo ASR com variantes de arquitetura"
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        required=True,
        help="Nome da variante do modelo a ser usada (ex: ConformerASRBase, ConformerASRPE, ConformerASRDA, TransformerASR_DA)",
    )
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default="./datasets",
        help="Diretório dos splits do dataset",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="./data/processed",
        help="Diretório dos dados processados",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="./models",
        help="Diretório para salvar os modelos",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="transformer_da",
        help="Nome do modelo para salvamento",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--dim_model", type=int, default=512, help="Dimensão do modelo")
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Número de cabeças de atenção"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Número de camadas da arquitetura"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Taxa de aprendizado")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser.add_argument(
        "--patience", type=int, default=10, help="Pacência para early stopping"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Dispositivo (cuda ou cpu)"
    )
    parser.add_argument(
        "--input_size", type=int, default=128, help="Tamanho da entrada"
    )
    args = parser.parse_args()
    main(args)
