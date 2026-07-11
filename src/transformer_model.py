import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len=512):

        super().__init__()

        pe = torch.zeros(max_len, embed_dim)

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(

            torch.arange(
                0,
                embed_dim,
                2
            ).float()

            * (-math.log(10000.0) / embed_dim)

        )

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


class MedicalTransformer(nn.Module):

    def __init__(

        self,

        vocab_size,

        embed_dim=256,

        num_heads=8,

        num_layers=6,

        ff_dim=1024,

        max_len=64,

        dropout=0.1

    ):

        super().__init__()

        self.embedding = nn.Embedding(

            vocab_size,

            embed_dim,

            padding_idx=0

        )

        self.position = PositionalEncoding(

            embed_dim,

            max_len

        )

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=embed_dim,

            nhead=num_heads,

            dim_feedforward=ff_dim,

            dropout=dropout,

            batch_first=True,

            activation="gelu"

        )

        self.encoder = nn.TransformerEncoder(

            encoder_layer,

            num_layers=num_layers

        )

        self.dropout = nn.Dropout(dropout)

        # Language Modeling Head

        self.fc = nn.Linear(

            embed_dim,

            vocab_size

        )

        self._init_weights()

    def _init_weights(self):

        for module in self.modules():

            if isinstance(module, nn.Linear):

                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:

                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):

                nn.init.normal_(

                    module.weight,

                    mean=0,

                    std=0.02

                )

    def forward(self, input_ids):

        x = self.embedding(input_ids)

        x = self.position(x)

        x = self.dropout(x)

        encoded = self.encoder(x)

        # Sentence Embedding

        sentence_embedding = encoded.mean(dim=1)

        # Token Prediction

        logits = self.fc(encoded)

        return logits, sentence_embedding