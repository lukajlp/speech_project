import math
import torch
import torch.nn as nn
import torchaudio.transforms as transforms


# ========================
# Positional Encoding (única definição)
# ========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, time, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ========================
# Arquiteturas Transformer
# ========================
class TransformerASR(nn.Module):
    def __init__(self, vocab_size, dim_model=512, num_heads=8):
        super().__init__()
        self.embedding = nn.Linear(128, dim_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model, nhead=num_heads, batch_first=True
            ),
            num_layers=6,
        )
        self.fc = nn.Linear(dim_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return x.log_softmax(dim=-1)


class TransformerASR_PE(nn.Module):
    def __init__(
        self, input_size=128, vocab_size=30, dim_model=512, num_heads=8, num_layers=4
    ):
        super().__init__()
        self.embedding = nn.Linear(input_size, dim_model)
        self.pos_encoder = PositionalEncoding(dim_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(dim_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return x.log_softmax(dim=-1)


class TransformerASR_DA(nn.Module):
    def __init__(
        self, input_size=128, vocab_size=30, dim_model=512, num_heads=8, num_layers=2
    ):
        super().__init__()
        self.embedding = nn.Linear(input_size, dim_model)
        self.pos_encoder = PositionalEncoding(dim_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(dim_model, vocab_size)
        # Data augmentation: SpecAugment aplicado no espaço de embedding
        self.freq_mask = transforms.FrequencyMasking(freq_mask_param=27)
        self.time_mask = transforms.TimeMasking(time_mask_param=100)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        if self.training:
            x = self.freq_mask(x)
            x = self.time_mask(x)
        x = self.transformer(x)
        return x.log_softmax(dim=-1)


# ========================
# Bloco Conformer
# ========================
class ConformerBlock(nn.Module):
    def __init__(self, d_model, nhead, ff_multiplier=4, dropout=0.1):
        super().__init__()
        # Primeira subcamada Feed-Forward
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_multiplier * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_multiplier * d_model, d_model),
            nn.Dropout(dropout),
        )
        # Multi-head Self-Attention
        self.mha = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # Módulo Convolucional
        self.conv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Conv1d(
                d_model, d_model, kernel_size=3, padding=1, groups=d_model
            ),  # Convolução depthwise
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )
        # Segunda subcamada Feed-Forward
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_multiplier * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_multiplier * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, time, d_model)
        x = x + 0.5 * self.ff1(x)
        attn_output, _ = self.mha(x, x, x)
        x = x + attn_output
        conv_input = x.transpose(1, 2)  # Transforma para (batch, d_model, time)
        conv_output = self.conv(conv_input)
        conv_output = conv_output.transpose(1, 2)  # Volta para (batch, time, d_model)
        x = x + conv_output
        x = x + 0.5 * self.ff2(x)
        x = self.norm(x)
        return x


# ========================
# Variantes do Conformer
# ========================


# Variante 1: ConformerASRBase - Versão básica sem positional encoding explícito
class ConformerASRBase(nn.Module):
    def __init__(
        self,
        vocab_size,
        input_size=128,
        dim_model=256,
        num_heads=4,
        num_layers=6,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = nn.Linear(input_size, dim_model)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(dim_model, num_heads, ff_multiplier=4, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(dim_model, vocab_size)

    def forward(self, x):
        # x: (batch, time, features)
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.fc(x)
        return x.log_softmax(dim=-1)


# Variante 2: ConformerASRPE - Com positional encoding explícito
class ConformerASRPE(nn.Module):
    def __init__(
        self,
        vocab_size,
        input_size=128,
        dim_model=256,
        num_heads=4,
        num_layers=6,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = nn.Linear(input_size, dim_model)
        self.pos_encoder = PositionalEncoding(dim_model, dropout=dropout)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(dim_model, num_heads, ff_multiplier=4, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(dim_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        x = self.fc(x)
        return x.log_softmax(dim=-1)


# Variante 3: ConformerASRDA - Com Data Augmentation (SpecAugment) aplicado durante o treinamento
class ConformerASRDA(nn.Module):
    def __init__(
        self,
        vocab_size,
        input_size=128,
        dim_model=256,
        num_heads=4,
        num_layers=6,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = nn.Linear(input_size, dim_model)
        self.pos_encoder = PositionalEncoding(dim_model, dropout=dropout)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(dim_model, num_heads, ff_multiplier=4, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(dim_model, vocab_size)
        # Data augmentation: SpecAugment
        self.freq_mask = transforms.FrequencyMasking(freq_mask_param=27)
        self.time_mask = transforms.TimeMasking(time_mask_param=100)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        if self.training:
            x = self.freq_mask(x)
            x = self.time_mask(x)
        for block in self.blocks:
            x = block(x)
        x = self.fc(x)
        return x.log_softmax(dim=-1)
