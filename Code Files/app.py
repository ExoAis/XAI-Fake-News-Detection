from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import os
import h5py
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# LNN
import torch
import torch.nn as nn
from LiqudNet.liquidnet.main import LiquidNet
from torch.utils.data import random_split, DataLoader

#KAN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from tqdm.auto import tqdm
import numpy as np

import torch
import torch.nn.functional as F
import math

from torchsummary import summary

# Preprocessing
import re
import nltk
import spacy
import fasttext
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import html
from lime.lime_text import LimeTextExplainer
from captum.attr import Saliency, IntegratedGradients

SECRET_API_KEY = "your-secret-api-key"


class CharacterEmbeddingsReductionModel(nn.Module):
    def __init__(self, input_size = 128, hidden_units = 256, output_size = 1, num_heads = 4):
        super(CharacterEmbeddingsReductionModel, self).__init__()

        self.mish = nn.Mish()

        # CNN Block for Feature Extraction
        self.conv1 = nn.Conv1d(in_channels = input_size, out_channels = hidden_units // 2, kernel_size = 3, padding = 1).to("cuda" if torch.cuda.is_available() else "cpu")
        self.conv2 = nn.Conv1d(in_channels = hidden_units // 2, out_channels = hidden_units, kernel_size = 3, padding = 1).to("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_norm = nn.BatchNorm1d(hidden_units).to("cuda" if torch.cuda.is_available() else "cpu")

        # GRU Layers for Temporal Encoding
        self.gru_1 = nn.GRU(hidden_units, hidden_units // 4, num_layers = 1, bidirectional = True, batch_first = True).to("cuda" if torch.cuda.is_available() else "cpu")
        self.norm_1 = nn.LayerNorm(hidden_units // 2).to("cuda" if torch.cuda.is_available() else "cpu")
        self.gru_2 = nn.GRU(hidden_units // 2, hidden_units // 64, num_layers = 1, bidirectional = False, batch_first = True).to("cuda" if torch.cuda.is_available() else "cpu")
        self.norm_2 = nn.LayerNorm(hidden_units // 64).to("cuda" if torch.cuda.is_available() else "cpu")
        self.gru_3 = nn.GRU(hidden_units // 4, hidden_units // 16, num_layers = 1, bidirectional = True, batch_first = True).to("cuda" if torch.cuda.is_available() else "cpu")
        self.norm_3 = nn.LayerNorm(hidden_units // 8).to("cuda" if torch.cuda.is_available() else "cpu")
        self.gru_4 = nn.GRU(hidden_units // 8, hidden_units // 64, num_layers = 1, bidirectional = False, batch_first = True).to("cuda" if torch.cuda.is_available() else "cpu")
        self.norm_4 = nn.LayerNorm(hidden_units // 64).to("cuda" if torch.cuda.is_available() else "cpu")

        # Self-Attention Layer
        self.attention = nn.MultiheadAttention(embed_dim = hidden_units // 64, num_heads = num_heads, batch_first = True).to("cuda" if torch.cuda.is_available() else "cpu")

        # Final Layers
        self.o_1 = nn.Linear(125 * hidden_units // 64, hidden_units).to("cuda" if torch.cuda.is_available() else "cpu")
        self.o_2 = nn.Linear(hidden_units, hidden_units // 2).to("cuda" if torch.cuda.is_available() else "cpu")
        self.o_3 = nn.Linear(hidden_units // 2, hidden_units // 4).to("cuda" if torch.cuda.is_available() else "cpu")
        self.o_4 = nn.Linear(hidden_units // 4, hidden_units // 8).to("cuda" if torch.cuda.is_available() else "cpu")
        self.o_5 = nn.Linear(hidden_units // 8, output_size).to("cuda" if torch.cuda.is_available() else "cpu")

        # Dropout for Regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        batch_size = x.shape[0]

        # CNN Feature Extraction
        x = x.permute(0, 2, 1)  # Change to (batch, channels, seq_len)
        x = self.mish(self.conv1(x))
        x = self.mish(self.conv2(x))
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)  # Change back to (batch, seq_len, channels)

        # GRU Layers
        outputs, _ = self.gru_1(x)
        outputs = self.norm_1(outputs)
        outputs, _ = self.gru_2(outputs)
        outputs = self.norm_2(outputs)
        reduced_character_embeddings = outputs.reshape(batch_size, 125, -1)
        outputs, _ = self.gru_3(outputs.reshape(batch_size, 125, -1))
        outputs = self.norm_3(outputs)
        outputs, _ = self.gru_4(outputs)
        outputs = self.norm_4(outputs)

        # Self-Attention
        outputs, _ = self.attention(outputs, outputs, outputs)

        # Fully Connected Layers
        outputs = self.mish(self.o_1(outputs.reshape(batch_size, -1)))
        outputs = self.dropout(outputs)
        outputs = self.mish(self.o_2(outputs))
        outputs = self.dropout(outputs)
        outputs = self.mish(self.o_3(outputs))
        outputs = self.mish(self.o_4(outputs))
        outputs = self.o_5(outputs)

        return outputs, reduced_character_embeddings
    
class FakeNewsDetectionModel(nn.Module):
    def __init__(self, input_size, hidden_units, output_size):
        super(FakeNewsDetectionModel, self).__init__()
        # Word Embeddings Block + Positional Embeddings Block
        self.multihead_attn_wepe = nn.MultiheadAttention(input_size, 4, batch_first = True).to("cuda" if torch.cuda.is_available() else "cpu")
        self.gru_1_wepe = nn.GRU(input_size, hidden_units // 4, num_layers = 1, bidirectional = True, batch_first = True).to("cuda" if torch.cuda.is_available() else "cpu")
        self.gru_2_wepe = nn.GRU(hidden_units // 2, hidden_units // 16, num_layers = 1, bidirectional = False, batch_first = True).to("cuda" if torch.cuda.is_available() else "cpu")

        # Character Embeddings Block
        self.conv1_ce = nn.Conv1d(hidden_units, hidden_units // 2, kernel_size = 3, stride = 1, padding = 1).to("cuda" if torch.cuda.is_available() else "cpu")
        self.conv2_ce = nn.Conv1d(hidden_units // 2, hidden_units // 4, kernel_size = 6, stride = 1, padding = 0).to("cuda" if torch.cuda.is_available() else "cpu")
        self.gru_1_ce = nn.GRU(hidden_units // 4, hidden_units // 16, num_layers = 1, bidirectional = True, batch_first = True).to("cuda" if torch.cuda.is_available() else "cpu")
        self.gru_2_ce = nn.GRU(hidden_units // 8 , hidden_units // 16, num_layers = 1, bidirectional = False, batch_first = True).to("cuda" if torch.cuda.is_available() else "cpu")

        # Output and Shared Block
        self.fc = nn.Linear(120 * 2 * hidden_units // 16, hidden_units * 4).to("cuda" if torch.cuda.is_available() else "cpu")
        self.ltc_cell = LiquidNet(hidden_units * 2).to("cuda" if torch.cuda.is_available() else "cpu")  # LiquidNet cell
        self.kan_1 = KANLinear(hidden_units * 2, hidden_units, base_activation = nn.SiLU).to("cuda" if torch.cuda.is_available() else "cpu") # KAN
        self.kan_2 = KANLinear(hidden_units, hidden_units // 2, base_activation = nn.SiLU).to("cuda" if torch.cuda.is_available() else "cpu")
        self.o = nn.Linear(hidden_units // 2, output_size).to("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, y, z, return_attention = False):
        # LiquidNet requires an initial hidden state
        batch_size = x.size(0)
        hidden_state = torch.zeros(batch_size, self.ltc_cell._num_units).to(x.device)

        # Word Embeddings Pass + Positional Embeddings Pass
        outputs = torch.add(x, y)
        outputs, attn_weights = self.multihead_attn_wepe(outputs, outputs, outputs, need_weights = True, average_attn_weights = False)  # Get attention weights
        outputs, _ = self.gru_1_wepe(outputs)
        outputs_wepe, _ = self.gru_2_wepe(outputs)

        # Character Embeddings Pass
        outputs = self.conv1_ce(z.reshape(batch_size, 64, 125))
        outputs = self.conv2_ce(outputs)
        outputs, _ = self.gru_1_ce(outputs.reshape(batch_size, 120, 16))
        outputs_ce, _ = self.gru_2_ce(outputs)

        # Pass through Output and Shared Block
        outputs = torch.cat((outputs_wepe, outputs_ce), dim = 2)
        outputs = self.fc(outputs.reshape(batch_size, 120 * hidden_units // 8))
        outputs, _ = self.ltc_cell(outputs, hidden_state)
        outputs = self.kan_1(outputs)
        outputs = self.kan_2(outputs)
        outputs = self.o(outputs)

        if return_attention:
            return outputs, attn_weights  # Return attention weights for explainability

        return outputs
    
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size = 5,
        spline_order = 3,
        scale_noise = 0.1,
        scale_base = 1.0,
        scale_spline = 1.0,
        enable_standalone_scale_spline = True,
        base_activation = torch.nn.SiLU,
        grid_eps = 0.02,
        grid_range = [-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a = math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a = math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin = 0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim = 0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device = x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype = torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device = x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device = x.device).unsqueeze(1),
            ],
            dim = 0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )
    
class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size = 5,
        spline_order = 3,
        scale_noise = 0.1,
        scale_base = 1.0,
        scale_spline = 1.0,
        base_activation = torch.nn.SiLU,
        grid_eps = 0.02,
        grid_range = [-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size = grid_size,
                    spline_order = spline_order,
                    scale_noise = scale_noise,
                    scale_base = scale_base,
                    scale_spline = scale_spline,
                    base_activation = base_activation,
                    grid_eps = grid_eps,
                    grid_range = grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid = False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation = 1.0, regularize_entropy = 1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

app = Flask(__name__)

# Initialize the model
input_size = 128  # Input size matches your embeddings
hidden_units = 64  # Number of units for LiquidNet
output_size = 1  # Binary classification (real/fake)
model = FakeNewsDetectionModel(input_size, hidden_units, output_size)

# Load Character Embeddings Reduction Model

input_size = 128  # Input size matches your embeddings
hidden_units_character = 256  # Number of units for LiquidNet
output_size = 1  # Binary classification (real/fake)
character_reduction_model = CharacterEmbeddingsReductionModel(input_size, hidden_units_character, output_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
character_reduction_model = character_reduction_model.to(device)
character_reduction_model.load_state_dict(torch.load('Saved/Model_Character_Reduction_125_64.pth', map_location = device))

# Load Fake News Classification Model

fake_new_model = torch.load(
    'Saved/Model_Combined_Word_Positional_Character.pth',
    map_location=torch.device('cpu')  # Force loading on CPU
)

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Load FastText Model
model = fasttext.load_model('unsupervised_fasttext_model.bin')

# Set Parameters
MAX_WORDS = 120
MAX_CHARS = 2000
EMBEDDING_DIM = 128

# Generate Positional Encoding
positional_encoding = np.zeros((MAX_WORDS, EMBEDDING_DIM))
position = np.arange(MAX_WORDS)[:, np.newaxis]
div_term = np.exp(np.arange(0, EMBEDDING_DIM, 2) * -(np.log(10000.0) / EMBEDDING_DIM))
positional_encoding[:, 0::2] = np.sin(position * div_term)
positional_encoding[:, 1::2] = np.cos(position * div_term)
positional_encoding = positional_encoding.astype(np.float16)

def preprocess_and_generate_embedding(text):
    """
    Preprocesses text and generates word, positional, and character embeddings.
    Returns:
        word_embedding (np.array): (MAX_WORDS, EMBEDDING_DIM)
        positional_embedding (np.array): (MAX_WORDS, EMBEDDING_DIM)
        char_embedding (np.array): (MAX_CHARS, EMBEDDING_DIM)
    """
    # 1️⃣ Text Standardization
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'http\S+|www\S+|@\S+', '', text)

    # 2️⃣ Tokenization
    tokens = word_tokenize(text)

    # 3️⃣ Stopword Removal
    tokens = [word for word in tokens if word not in stop_words]

    # 4️⃣ Lemmatization
    doc = nlp(' '.join(tokens))
    words = [token.lemma_ for token in doc]

    # 5️⃣ Generate Word Embeddings
    words = words[:MAX_WORDS] if len(words) > MAX_WORDS else words + [''] * (MAX_WORDS - len(words))
    vectors = [model.get_word_vector(word) if word in model.words else np.zeros(EMBEDDING_DIM) for word in words]
    word_embedding = np.vstack(vectors)

    # 6️⃣ Generate Positional Encoding
    positional_embedding = positional_encoding[:len(words)]

    # 7️⃣ Generate Character Embeddings
    chars = list(text)[:MAX_CHARS] if len(text) > MAX_CHARS else list(text) + [' '] * (MAX_CHARS - len(text))
    char_vectors = [model.get_word_vector(char) if char in model.words else np.zeros(EMBEDDING_DIM) for char in chars]
    char_embedding = np.vstack(char_vectors)

    return word_embedding.astype(np.float16), positional_embedding.astype(np.float16), char_embedding.astype(np.float16), words

def pipeline(news_article, character_model, news_model, device='cpu'):
    word_embeddings, positional_embeddings, character_embeddings = preprocess_and_generate_embedding(news_article)

    word_embeddings = torch.Tensor(word_embeddings).to(device)
    positional_embeddings = torch.Tensor(positional_embeddings).to(device)
    character_embeddings = torch.Tensor(character_embeddings).to(device)

    word_embeddings = word_embeddings[np.newaxis, :, :]
    positional_embeddings = positional_embeddings[np.newaxis, :, :]
    character_embeddings = character_embeddings[np.newaxis, :, :]

    _, character_embeddings_reduced = character_model(character_embeddings.float())
    output = news_model(word_embeddings.float(), positional_embeddings.float(), character_embeddings_reduced.float())
    output = nn.Sigmoid()(output)
    label = (output > 0.7).detach().cpu().numpy()
    accuracy = output.detach().cpu().numpy().flatten()[0] * 100
    if label == 0:
        accuracy = 100 - accuracy
    accuracy = "{:.2f}".format(int(accuracy * 100) / 100)
    if label == 1:
        decision = 'News verified, likely real' + ' with accuracy ' + str(accuracy) + '%'
    else:
        decision = 'news unverified, likely fake' + ' with accuracy ' + str(accuracy) + '%'

    return decision

class WordOnlyWrapper(nn.Module):
    def __init__(self, base_model, fixed_pos_emb, fixed_char_reduced):
        super().__init__()
        self.base_model = base_model.train()
        self.pos_emb = fixed_pos_emb
        self.char_reduced = fixed_char_reduced

    def forward(self, word_emb):
        self.base_model.train()  # Required for CuDNN backward through RNNs
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.pos_emb = self.pos_emb.repeat(50, 1, 1).requires_grad_()
        self.char_reduced = self.char_reduced.repeat(50, 1, 1).requires_grad_()
        return self.base_model(word_emb.requires_grad_(), self.pos_emb, self.char_reduced)

class WordOnlyWrapperSaliency(nn.Module):
    def __init__(self, base_model, fixed_pos_emb, fixed_char_reduced):
        super().__init__()
        self.base_model = base_model.train()
        self.pos_emb = fixed_pos_emb
        self.char_reduced = fixed_char_reduced

    def forward(self, word_emb):
        self.base_model.train()  # Required for CuDNN backward through RNNs
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.pos_emb = self.pos_emb.requires_grad_()
        self.char_reduced = self.char_reduced.requires_grad_()
        return self.base_model(word_emb.requires_grad_(), self.pos_emb, self.char_reduced)


def unified_pipeline(news_article, character_model, news_model, device='cpu'):
    news_model.eval()
    character_model.eval()

    # Preprocessing
    word_embeddings, positional_embeddings, character_embeddings, tokens = preprocess_and_generate_embedding(news_article)
    word_embeddings = torch.tensor(word_embeddings).unsqueeze(0).to(device)
    positional_embeddings = torch.tensor(positional_embeddings).unsqueeze(0).to(device)
    character_embeddings = torch.tensor(character_embeddings).unsqueeze(0).to(device)

    with torch.no_grad():
        _, char_reduced = character_model(character_embeddings.float())
        output, attention_weights = news_model(
            word_embeddings.float(),
            positional_embeddings.float(),
            char_reduced.float(),
            return_attention=True
        )

    # Classification result
    prob = torch.sigmoid(output).item()
    label = int(prob >= 0.5)  # Now label = 1 (fake) only if prob ≥ 0.5
    confidence = "{:.2f}".format((1 - prob) * 100 if label == 0 else prob * 100)
    decision = (
        f"News unverified, likely fake with confidence {confidence}%"
        if label else
        f"News verified, likely real with confidence {confidence}%"
    )

    # 1. Attention Weights
    attention_weights = attention_weights.squeeze(0).mean(0).mean(0)
    attention_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min())
    words = news_article.split()
    attention_map = [(word, float(attention_weights[i])) for i, word in enumerate(words[:len(attention_weights)])]
    explanation_attention = generate_attention_visualization(attention_map)

    # 2. LIME
    explainer = LimeTextExplainer(class_names=["Fake", "Real"], bow=False)
    lime_exp = explainer.explain_instance(
        news_article,
        classifier_fn=lambda x: predict_function(x, character_model, news_model, device),
        num_features=10,
        num_samples=10  # try 200 or 300 for a decent speedup
    )
    explanation_lime = lime_exp.as_list()

    #3 IG
    wrapped_model = WordOnlyWrapper(news_model, positional_embeddings.float(), char_reduced)
    wrapped_model.eval()
    ig = IntegratedGradients(wrapped_model)
    # word_embeddings shape: (1, 120, 128)
    attributions_ig = ig.attribute(word_embeddings.float(), target=0)
    # Convert to token-level scores
    explanation_ig = [(tokens[i], float(val))
                      for i, val in enumerate(attributions_ig[0].sum(dim=-1))]

    #4 Saliency
    wrapped_model = WordOnlyWrapperSaliency(news_model, positional_embeddings.float(), char_reduced)
    # Create Saliency object
    saliency = Saliency(wrapped_model)
    # Compute saliency attributions
    attributions_sal = saliency.attribute(word_embeddings.float(), target=0)
    # Token-level attribution scores (sum over embedding dim)
    explanation_sal = [(tokens[i], float(val))
                      for i, val in enumerate(attributions_sal[0].sum(dim=-1))]

    return {
        "decision": decision,
        "attention": explanation_attention,
        "lime": explanation_lime,
        "ig": explanation_ig,
        "saliency": explanation_sal
    }


def generate_attention_visualization(attention_map):
    html_output = ""
    for word, weight in attention_map:
        safe_word = html.escape(word)
        html_output += f'<span style="background-color:rgba(255, 0, 0, {weight:.2f}); padding:2px; margin:2px;">{safe_word}</span> '
    return html_output


def predict_function(texts, character_model, news_model, device='cpu'):
    word_tensors, pos_tensors, char_tensors = [], [], []
    for text in texts:
        word_embeddings, pos_embeddings, char_embeddings, _ = preprocess_and_generate_embedding(text)
        word_tensors.append(torch.tensor(word_embeddings))
        pos_tensors.append(torch.tensor(pos_embeddings))
        char_tensors.append(torch.tensor(char_embeddings))
    word_tensor = torch.stack(word_tensors).to(device).float()
    pos_tensor = torch.stack(pos_tensors).to(device).float()
    char_tensor = torch.stack(char_tensors).to(device).float()
    with torch.no_grad():
        _, char_reduced = character_model(char_tensor)
        output = torch.sigmoid(news_model(word_tensor, pos_tensor, char_reduced)).cpu().numpy()
    return np.hstack([(1 - output), output])

@app.route('/generate', methods=['POST'])
def generate():
    # Check for the x-api-key header
    api_key = request.headers.get('x-api-key')
    if api_key != SECRET_API_KEY:
        return jsonify({'error': 'Unauthorized'}), 401

    # Process the request
    data = request.get_json()
    article = data.get('article', '')
    if not article:
        return jsonify({'error': 'No article provided'}), 400

    explanation_dict = unified_pipeline(article, character_reduction_model, fake_new_model, device = 'cpu')
    
    return jsonify({
        'decision': explanation_dict["decision"],
        'attention': explanation_dict["attention"],
        'lime': explanation_dict["lime"],
        'ig': explanation_dict["ig"],
        'saliency': explanation_dict["saliency"]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)