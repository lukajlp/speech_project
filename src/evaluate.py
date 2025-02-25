import os
import torch
import argparse
from torch.utils.data import DataLoader
import jiwer

# Importa todas as variantes do modelo
from models import TransformerASR_DA, ConformerASRBase, ConformerASRPE, ConformerASRDA
from utils import collate_fn, decode_predictions


def evaluate_model(test_set_path, processed_dir, model_path, model_variant, device, dim_model, num_heads, num_layers):
    # Carrega o vocabulário
    vocab = torch.load(os.path.join(processed_dir, "vocab.pt"), weights_only=False)
    vocab_inv = {v: k for k, v in vocab.items()}

    # Mapeia os nomes das variantes para as classes correspondentes
    model_variants = {
        "TransformerASR_DA": TransformerASR_DA,
        "ConformerASRBase": ConformerASRBase,
        "ConformerASRPE": ConformerASRPE,
        "ConformerASRDA": ConformerASRDA,
    }
    if model_variant not in model_variants:
        raise ValueError(
            f"Modelo variante '{model_variant}' não suportado. Escolha entre: {list(model_variants.keys())}"
        )
    ModelClass = model_variants[model_variant]

    # Instancia o modelo com os hiperparâmetros informados
    model = ModelClass(vocab_size=len(vocab), dim_model=dim_model, num_heads=num_heads, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.to(device)
    model.eval()

    # Carrega o conjunto de teste
    test_set = torch.load(test_set_path, weights_only=False)
    test_loader = DataLoader(test_set, batch_size=16, collate_fn=collate_fn)

    all_references = []
    all_predictions = []

    with torch.no_grad():
        for waveforms, transcripts, _ in test_loader:
            waveforms = waveforms.to(device)
            logits = model(waveforms)
            pred_texts = decode_predictions(logits, vocab)
            # Converte as transcrições verdadeiras de índices para texto
            true_texts = []
            for seq in transcripts:
                text = []
                for idx in seq:
                    if idx.item() in vocab_inv and idx.item() not in [0, 1]:
                        text.append(vocab_inv[idx.item()])
                true_texts.append("".join(text))
            all_references.extend(true_texts)
            all_predictions.extend(pred_texts)

    wer_score = jiwer.wer(all_references, all_predictions)
    print(f"WER: {wer_score * 100:.2f}%")
    print("\nExemplos de predições:")
    for i in range(min(5, len(all_references))):
        print(f"Real: {all_references[i]}")
        print(f"Previsto: {all_predictions[i]}\n")


def main(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    evaluate_model(
        args.test_set_path,
        args.processed_dir,
        args.model_path,
        args.model_variant,
        device,
        args.dim_model,
        args.num_heads,
        args.num_layers
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação do modelo ASR")
    parser.add_argument(
        "--test_set_path",
        type=str,
        default="./datasets/test_set.pt",
        help="Caminho para o conjunto de teste",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="./data/processed",
        help="Diretório dos dados processados",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/transformer_da/best_model.pt",
        help="Caminho do modelo salvo",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default="TransformerASR_DA",
        help="Nome da variante do modelo a ser usada (ex: TransformerASR_DA, ConformerASRBase, ConformerASRPE, ConformerASRDA)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Dispositivo (cuda ou cpu)"
    )
    # Parâmetros do modelo que devem ser compatíveis com o treinamento
    parser.add_argument("--dim_model", type=int, default=144, help="Dimensão do modelo")
    parser.add_argument("--num_heads", type=int, default=4, help="Número de cabeças de atenção")
    parser.add_argument("--num_layers", type=int, default=16, help="Número de camadas do encoder")
    args = parser.parse_args()
    main(args)
