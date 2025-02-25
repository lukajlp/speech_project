import torch

def main():
    # Definir os caminhos corretos para os arquivos
    dataset_path = "../data/processed/full_dataset.pt"
    vocab_path = "../data/processed/vocab.pt"
    
    # Carregar o dataset pré-processado
    dataset = torch.load(dataset_path)
    
    # Selecionar uma amostra (por exemplo, a primeira)
    sample_idx = 0
    waveform, transcript_indices = dataset[sample_idx]
    
    print("🔍 Exemplo de amostra pré-processada:")
    print(f"- Formato do espectrograma: {waveform.shape} (Canais, Frequências, Tempo)")
    print(f"- Transcrição (índices): {transcript_indices}")
    print(f"- Comprimento da transcrição: {len(transcript_indices)}")
    
    # Carregar o vocabulário e criar um mapeamento inverso
    vocab = torch.load(vocab_path)
    vocab_inv = {v: k for k, v in vocab.items()}
    
    # Decodificar a transcrição, ignorando os índices 0 e 1
    decoded_text = ''.join(
        [vocab_inv[idx.item()] if hasattr(idx, "item") else vocab_inv[idx]
         for idx in transcript_indices if idx not in [0, 1]]
    )
    print(f"- Transcrição decodificada: '{decoded_text}'")

if __name__ == "__main__":
    main()
