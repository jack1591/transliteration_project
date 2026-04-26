from fastapi import FastAPI
import torch
from torch import nn
import math
import json
from pydantic import BaseModel
from fastapi.responses import FileResponse

MODEL_PATH = './transliteration_model_weights.pth'
SRC_VOCAB_INFO_PATH = './src_vocab_info.json'
TRG_VOCAB_INFO_PATH = './trg_vocab_info.json'

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads: int = 6, embed_dim: int = 256):
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.q = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.k = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.v = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.out = nn.Linear(in_features=embed_dim, out_features=embed_dim)

        self.scale = self.head_dim ** 0.5

    def forward(self, query, key, value, mask=None):
        B, T, D = query.shape

        Q = self.q(query)
        K = self.k(key)
        V = self.v(value)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, HEADS, T, HEAD_DIM)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, HEADS, T, HEAD_DIM)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, HEADS, T, HEAD_DIM)

        qk_dot = torch.matmul(Q, K.transpose(-1, -2)) / self.scale

        if mask is not None:
            qk_dot = qk_dot.masked_fill(mask == 0, float('-inf'))
        attention_scores = torch.softmax(qk_dot, dim=-1)

        head_out = torch.matmul(attention_scores, V)
        # contiguous - соединяем в памяти все числа в одно место - потом в view склеиваем
        head_out = head_out.transpose(1, 2).contiguous().view(B, T, D)

        return self.out(head_out)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int = 256, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                             (- math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int = 256, dim_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_features = embed_dim, out_features = dim_ff)
        self.linear2 = nn.Linear(in_features = dim_ff, out_features = embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, embed_dim: int = 256, dim_ff: int = 1024, num_heads: int = 6, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(num_heads, embed_dim)
        self.ff = FeedForward(embed_dim, dim_ff, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask = None):
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x

class Decoder(nn.Module):
    def __init__(self, embed_dim: int = 256, dim_ff: int = 1024, num_heads: int = 6, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(num_heads, embed_dim)
        self.cross_attn = MultiHeadSelfAttention(num_heads, embed_dim)
        self.ff = FeedForward(embed_dim, dim_ff, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    def forward(self, x, enc_out, src_mask = None, tgt_mask = None):
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        cross_attn_out = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout3(ff_out))
        return x


class MyTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 embed_dim: int = 256, dim_ff: int = 1024, dropout: float = 0.1,
                 num_heads: int = 8, num_layers: int = 4, pad_idx: int = 0, max_len: int = 1000):
        super().__init__()
        self.pad_idx = pad_idx

        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoding(embed_dim, max_len, dropout)

        self.encoder_layers = nn.ModuleList([
            Encoder(embed_dim, dim_ff, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            Decoder(embed_dim, dim_ff, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(in_features=embed_dim, out_features=tgt_vocab_size)
        self.scale = embed_dim ** 0.5

    def make_causal_mask(self, size, device):
        # не дает decoder'у смотреть в будущее
        mask = torch.tril(torch.ones(size, size, device=device)).bool().unsqueeze(0).unsqueeze(0)
        return mask  # (1, 1, T, T)

    def make_pad_mask(self, seq):
        # не дает attention обращать внимание на padding-токены
        return (seq != self.pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)

    def forward(self, src, tgt):
        src_mask = self.make_pad_mask(src)
        tgt_mask = self.make_pad_mask(tgt) & self.make_causal_mask(tgt.size(1), tgt.device)

        # Encoder
        src_emb = self.pos_embedding(self.src_embedding(src) * self.scale)
        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)

        tgt_emb = self.pos_embedding(self.tgt_embedding(tgt) * self.scale)
        dec_out = tgt_emb
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)

        # проекция на словарь
        return self.fc_out(dec_out)

def get_vocab_info(vocab_path):
    data = {}
    with open(vocab_path, 'r', encoding = 'utf-8') as f:
        data = json.load(f)
    bos_idx = data['bos']
    eos_idx = data['eos']
    pad_idx = data['pad']
    vocab_size = data['vocab_len']
    vocab_idxs = data['vocab_idxs']
    return bos_idx, eos_idx, pad_idx, vocab_size, vocab_idxs

def load_model(model_path: str, src_vocab_size: int, trg_vocab_size: int, device: torch.device):
    model = MyTransformer(src_vocab_size, trg_vocab_size)
    state_dict = torch.load(model_path, map_location = device)
    model.load_state_dict(state_dict)
    return model


class VocabWrapper:
    """Обёртка над JSON vocab_idxs: даёт методы char2idx / idx2char / encode / decode."""

    def __init__(self, vocab_info_path: str):
        with open(vocab_info_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.bos_idx = data["bos"]
        self.eos_idx = data["eos"]
        self.pad_idx = data["pad"]
        self.vocab_size = data["vocab_len"]
        # vocab_idxs: {"a": 4, "b": 5, ...}  или  {"AA0": 4, ...}
        self._char2idx = {str(k): int(v) for k, v in data["vocab_idxs"].items()}
        self._idx2char = {int(v): str(k) for k, v in data["vocab_idxs"].items()}

    def char2idx(self, ch: str) -> int:
        return self._char2idx.get(ch, self._char2idx.get("<unk>", 3))

    def idx2char(self, idx: int) -> str:
        return self._idx2char.get(idx, "<unk>")

    def get_bos(self) -> int:
        return self.bos_idx

    def get_eos(self) -> int:
        return self.eos_idx

    def get_pad(self) -> int:
        return self.pad_idx

    def encode(self, seq):
        """Для src: seq = list of chars, для tgt: seq = list of phoneme strings."""
        return [self.bos_idx] + [self.char2idx(ch) for ch in seq] + [self.eos_idx]

    def decode(self, indices):
        tokens = []
        for idx in indices:
            if idx == self.eos_idx:
                break
            if idx in (self.bos_idx, self.pad_idx):
                continue
            tokens.append(self.idx2char(idx))
        return tokens

import re
import torch

def transcribe_g2p(model, src_vocab, tgt_vocab, input_string, max_len=50, device=None):
    """
    model: обученный MyTransformer
    src_vocab: Vocabulary для графем
    tgt_vocab: PhonemeVocabulary для фонем
    input_string: слово для транскрибирования
    max_len: максимальная длина выходной последовательности
    device: если None, берётся из параметров модели (авто)
    Возвращает строку фонем через пробел.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    graphemes = list(input_string)
    src_indices = [src_vocab.char2idx(ch) for ch in graphemes]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)  # (1, src_len)

    bos_idx = tgt_vocab.get_bos()
    eos_idx = tgt_vocab.get_eos()

    tgt_indices = [bos_idx]

    for _ in range(max_len):
        tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(src_tensor, tgt_tensor)
            next_token_logits = logits[0, -1, :]
            next_token = next_token_logits.argmax(dim=-1).item()

        tgt_indices.append(next_token)
        if next_token == eos_idx:
            break

    phoneme_list = tgt_vocab.decode(tgt_indices)
    return ' '.join(phoneme_list)

def transcribe_text(
    model,
    src_vocab,
    tgt_vocab,
    text: str,
    separator: str = " | ",
    max_len: int = 50,
    device=None,
) -> str:
    words = re.findall(r"[a-zA-Z]+", text)

    if not words:
        return "", []

    word_results = []
    for word in words:
        ph = transcribe_g2p(model, src_vocab, tgt_vocab, word, max_len, device)
        word_results.append({"word": word, "phonemes": ph})

    full_phonemes = separator.join(r["phonemes"] for r in word_results)
    return full_phonemes, word_results

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
src_vocab = VocabWrapper(SRC_VOCAB_INFO_PATH)
tgt_vocab = VocabWrapper(TRG_VOCAB_INFO_PATH)

model = MyTransformer(src_vocab.vocab_size, tgt_vocab.vocab_size)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

app = FastAPI(title = 'G2P_transliteration')

@app.get("/")
async def index():
    return FileResponse("static/index.html")

class TransliterationRequest(BaseModel):
    text: str
    separator: str = " | "


class TransliterationResponse(BaseModel):
    text: str
    phonemes: str
    words: list[dict]  # [{word: "hello", phonemes: "HH AH0 L OW1"}, ...]


@app.post("/transliterate")
def transliterate(req: TransliterationRequest) -> TransliterationResponse:
    full_phonemes, word_results = transcribe_text(
        model, src_vocab, tgt_vocab, req.text, req.separator
    )
    return TransliterationResponse(
        text=req.text,
        phonemes=full_phonemes,
        words=word_results,
    )