# 🎙️ G2P Transliterator

Веб-приложение для автоматической фонетической транскрипции английских слов. В основе — собственная реализация архитектуры Transformer (encoder-decoder), обученная на датасете CMUdict.

Введите слово или предложение → получите транскрипцию в формате [ARPAbet](https://en.wikipedia.org/wiki/ARPABET).

```
cucumber → K Y UW1 K AH0 M B ER0
hello world → HH AH0 L OW1 | W ER1 L D
```

---
## Демонстрация
Интерфейс выполнен в стиле переводчика: в левом окне вы пишите предложения на английском. В правом окне - транслитерация каждого слова в составе предложений с дополнительной информацией о фонемах.

### Начальный экран
<img width="1873" height="851" alt="image" src="https://github.com/user-attachments/assets/f03cbdac-ee2e-4583-ad1e-1bf928bc3c9e" />

### Пример транскрипции
<img width="1848" height="859" alt="image" src="https://github.com/user-attachments/assets/38206f5f-1e22-4423-9340-66aba690d619" />


## Архитектура модели

Transformer реализован с нуля на PyTorch по статье [Attention Is All You Need](https://arxiv.org/abs/1706.03762):

| Параметр | Значение |
|---|---|
| Encoder layers | 4 |
| Decoder layers | 4 |
| Attention heads | 8 |
| Embedding dim | 256 |
| FFN dim | 1024 |
| Dropout | 0.1 |
| Source vocab | 78 (буквы, цифры, пунктуация) |
| Target vocab | 88 (ARPAbet фонемы) |

Модель включает: Multi-Head Self-Attention, Cross-Attention, Positional Encoding (синусоидальный), causal mask для декодера и padding mask.

Инференс — авторегрессивный (greedy decoding): на каждом шаге декодер генерирует следующую фонему, пока не выдаст `<eos>`.

## Данные

[CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) — 135 166 пар «слово → фонетическая транскрипция». Разбиение: 85% train / 15% validation.

## Обучение

- Оптимизатор: Adam, lr = 2e-4
- Loss: CrossEntropyLoss (ignore_index = pad)
- Gradient clipping: clip_grad_norm_
- Эпох: 15
- Платформа: Kaggle GPU (CUDA)

Полный код обучения — в `transliteration-transformer.ipynb`.

## Структура проекта

```
transliteration_project/
├── main.py                              # FastAPI backend + модель
├── static/
│   └── index.html                       # Веб-интерфейс
├── transliteration_model_weights.pth    # Веса модели (в .gitignore)
├── src_vocab_info.json                  # Словарь графем
├── trg_vocab_info.json                  # Словарь фонем
├── transliteration-transformer.ipynb    # Ноутбук с обучением
├── requirements.txt
└── .gitignore
```

## Запуск

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Веса модели

Скачайте файл `transliteration_model_weights.pth` и положите в корень проекта. Файл не включён в репозиторий из-за размера.

### 3. Запуск сервера

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Откройте [http://localhost:8000](http://localhost:8000) в браузере.

## API

### `POST /transliterate`

**Request:**
```json
{
  "text": "hello world",
  "separator": " | "
}
```

**Response:**
```json
{
  "text": "hello world",
  "phonemes": "HH AH0 L OW1 | W ER1 L D",
  "words": [
    {"word": "hello", "phonemes": "HH AH0 L OW1"},
    {"word": "world", "phonemes": "W ER1 L D"}
  ]
}
```

## Фонемы

Фонемы отображаются цветными чипами с категоризацией:

- 🔴 **Гласные** (vowels) — AA, AE, AH, EY, IY, OW...
- 🔵 **Смычные** (stops) — B, D, G, K, P, T
- 🟢 **Фрикативные** (fricatives) — F, S, SH, TH, V, Z...
- 🟡 **Носовые и плавные** (nasals & liquids) — L, M, N, R, W, Y

Цифры при гласных (0, 1, 2) обозначают уровень ударения.

## Стек

- **PyTorch** — модель и инференс
- **FastAPI** — REST API
- **HTML/CSS/JS** — фронтенд
