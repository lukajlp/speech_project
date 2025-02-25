import os
import argparse
import urllib.request
import tarfile


def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"Baixando {url} para {dest}...")
        urllib.request.urlretrieve(url, dest)
        print("Download concluído.")
    else:
        print(f"O arquivo {dest} já existe. Pulando download.")


def extract_tar(file_path, extract_path):
    print(f"Extraindo {file_path} para {extract_path}...")
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print("Extração concluída.")


def main(args):
    os.makedirs(args.raw_dir, exist_ok=True)
    test_clean_url = "https://www.openslr.org/resources/12/test-clean.tar.gz"
    train_clean_url = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"

    test_clean_dest = os.path.join(args.raw_dir, "test-clean.tar.gz")
    train_clean_dest = os.path.join(args.raw_dir, "train-clean-100.tar.gz")

    download_file(test_clean_url, test_clean_dest)
    download_file(train_clean_url, train_clean_dest)

    extract_tar(test_clean_dest, args.raw_dir)
    extract_tar(train_clean_dest, args.raw_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baixar e extrair o dataset LibriSpeech."
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="./data/raw",
        help="Diretório para armazenar os dados brutos",
    )
    args = parser.parse_args()
    main(args)
