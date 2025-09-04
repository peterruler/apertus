# Apertus-8B-Instruct-2509

Ein Python-Projekt zur Nutzung des Apertus-8B-Instruct-2509 Sprachmodells von swiss-ai.

## Überblick

Dieses Projekt demonstriert die Verwendung des Apertus-8B-Instruct-2509 Modells, einem fortschrittlichen Large Language Model (LLM), das von swiss-ai entwickelt wurde. Das Modell ist darauf spezialisiert, auf natürliche Weise auf Benutzeranfragen zu antworten und kann für verschiedene Textgenerierungsaufgaben eingesetzt werden.

## Funktionen

- Einfache Integration des Apertus-8B-Instruct-2509 Modells
- Unterstützung für GPU- und CPU-basierte Inferenz
- Beispielimplementierung für Textgenerierung
- Jupyter Notebook für interaktive Experimente

## Voraussetzungen

- Python 3.8 oder höher
- CUDA-kompatible GPU (empfohlen) oder CPU
- Mindestens 16GB RAM (32GB für optimale Performance)

## Installation

1. Klonen Sie das Repository:
```bash
git clone https://github.com/peterruler/apertus.git
cd apertus
```

2. Installieren Sie die erforderlichen Abhängigkeiten:
```bash
pip install -U transformers torch
```

Für GPU-Unterstützung (empfohlen):
```bash
pip install -U transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Schnellstart

### Jupyter Notebook

Das einfachste Einstieg ist über das bereitgestellte Jupyter Notebook:

1. Starten Sie Jupyter:
```bash
jupyter notebook
```

2. Öffnen Sie `apertus.ipynb` und führen Sie die Zellen aus.

### Google Colab

Sie können das Projekt auch direkt in Google Colab ausführen:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peterruler/apertus/blob/main/apertus.ipynb)

### Python Script

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Modell laden
model_name = "swiss-ai/Apertus-8B-Instruct-2509"
device = "cuda"  # für GPU oder "cpu" für CPU

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Prompt vorbereiten
prompt = "Erkläre mir die Schwerkraft in einfachen Worten."
messages = [{"role": "user", "content": prompt}]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# Inferenz
model_inputs = tokenizer([text], return_tensors="pt").to(device)
generated_ids = model.generate(**model_inputs, max_new_tokens=512)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
response = tokenizer.decode(output_ids, skip_special_tokens=True)

print(response)
```

## Verwendung

### Chat-Format

Das Modell erwartet Eingaben im Chat-Format:

```python
messages = [
    {"role": "user", "content": "Deine Frage hier"}
]
```

### Generierungsparameter

Sie können verschiedene Parameter anpassen:

```python
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,          # Maximale Anzahl neuer Token
    temperature=0.7,             # Kreativität (0.0-1.0)
    top_p=0.9,                   # Nucleus sampling
    do_sample=True,              # Aktiviert Sampling
    pad_token_id=tokenizer.eos_token_id
)
```

## Systemanforderungen

### Minimal
- RAM: 16GB
- VRAM: 8GB (für GPU-Nutzung)
- Speicher: 20GB

### Empfohlen
- RAM: 32GB oder mehr
- VRAM: 16GB oder mehr (RTX 4080/4090, A100, etc.)
- Speicher: 50GB

## Leistung

Das Modell bietet:
- Hohe Qualität bei Textgenerierung
- Unterstützung für multiple Sprachen
- Konsistente und kohärente Antworten
- Schnelle Inferenz bei ausreichender Hardware

## Fehlerbehebung

### Häufige Probleme

**CUDA out of memory:**
- Reduzieren Sie `max_new_tokens`
- Verwenden Sie `device="cpu"`
- Schließen Sie andere GPU-intensive Anwendungen

**Langsame Performance:**
- Stellen Sie sicher, dass CUDA korrekt installiert ist
- Verwenden Sie eine GPU mit ausreichend VRAM
- Optimieren Sie die Batch-Größe

**Modell lädt nicht:**
- Überprüfen Sie Ihre Internetverbindung
- Stellen Sie sicher, dass Sie Zugriff auf Hugging Face haben
- Verwenden Sie `pip install -U transformers`

## Beitragen

Contributions sind willkommen! Bitte:

1. Forken Sie das Repository
2. Erstellen Sie einen Feature-Branch
3. Committen Sie Ihre Änderungen
4. Erstellen Sie einen Pull Request

## Lizenz

Dieses Projekt folgt der Lizenz des ursprünglichen Apertus-8B-Instruct-2509 Modells. Weitere Details finden Sie auf der [Hugging Face Modellseite](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509).

## Links

- **Modell auf Hugging Face**: https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509
- **swiss-ai Organisation**: https://huggingface.co/swiss-ai
- **Transformers Dokumentation**: https://huggingface.co/docs/transformers

## Support

Für Fragen und Support:
- Öffnen Sie ein Issue in diesem Repository
- Konsultieren Sie die Hugging Face Dokumentation
- Besuchen Sie die swiss-ai Community