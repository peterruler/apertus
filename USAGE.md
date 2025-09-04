# Verwendungsanleitung

Diese Anleitung zeigt Ihnen, wie Sie das Apertus-8B-Instruct-2509 Modell effektiv nutzen können.

## Grundlegende Verwendung

### 1. Modell laden

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "swiss-ai/Apertus-8B-Instruct-2509"
device = "cuda"  # oder "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
```

### 2. Einfache Textgenerierung

```python
prompt = "Erkläre mir maschinelles Lernen."
messages = [{"role": "user", "content": prompt}]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

model_inputs = tokenizer([text], return_tensors="pt").to(device)
generated_ids = model.generate(**model_inputs, max_new_tokens=512)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
response = tokenizer.decode(output_ids, skip_special_tokens=True)
print(response)
```

## Erweiterte Konfiguration

### Generierungsparameter

```python
generation_config = {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "do_sample": True,
    "repetition_penalty": 1.1,
    "pad_token_id": tokenizer.eos_token_id
}

generated_ids = model.generate(**model_inputs, **generation_config)
```

### Batch-Verarbeitung

```python
prompts = [
    "Was ist künstliche Intelligenz?",
    "Erkläre Quantencomputing.",
    "Wie funktioniert das Internet?"
]

messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]
texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
         for msg in messages_batch]

model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
generated_ids = model.generate(**model_inputs, max_new_tokens=512)

# Antworten dekodieren
responses = []
for i, generated in enumerate(generated_ids):
    output_ids = generated[len(model_inputs.input_ids[i]):]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    responses.append(response)
```

## Chat-Konversationen

### Multi-Turn Dialog

```python
conversation = [
    {"role": "user", "content": "Hallo! Kannst du mir bei Python helfen?"},
]

# Erste Antwort
text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(device)
generated_ids = model.generate(**model_inputs, max_new_tokens=256)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
response1 = tokenizer.decode(output_ids, skip_special_tokens=True)

# Antwort zur Konversation hinzufügen
conversation.append({"role": "assistant", "content": response1})
conversation.append({"role": "user", "content": "Wie erstelle ich eine Liste in Python?"})

# Zweite Antwort
text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(device)
generated_ids = model.generate(**model_inputs, max_new_tokens=256)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
response2 = tokenizer.decode(output_ids, skip_special_tokens=True)

print("Antwort 1:", response1)
print("Antwort 2:", response2)
```

## Leistungsoptimierung

### Memory Management

```python
import torch
import gc

# Speicher freigeben
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Nach jeder Generierung
generated_ids = model.generate(**model_inputs, max_new_tokens=512)
clear_memory()
```

### Gradient Checkpointing

```python
# Für große Modelle mit begrenztem VRAM
model.gradient_checkpointing_enable()
```

### Float16 Precision

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
).to(device)
```

## Anwendungsbeispiele

### Code-Generierung

```python
prompt = """
Schreibe eine Python-Funktion, die eine Liste von Zahlen sortiert:

```python
def sort_numbers(numbers):
"""

messages = [{"role": "user", "content": prompt}]
# ... Generierung wie oben
```

### Textanalyse

```python
text_to_analyze = "Der Klimawandel ist eines der wichtigsten Themen unserer Zeit..."
prompt = f"Analysiere folgenden Text und fasse die Hauptpunkte zusammen:\n\n{text_to_analyze}"

messages = [{"role": "user", "content": prompt}]
# ... Generierung wie oben
```

### Kreatives Schreiben

```python
prompt = "Schreibe eine kurze Geschichte über einen Roboter, der lernt zu träumen."

messages = [{"role": "user", "content": prompt}]
# Höhere Temperatur für mehr Kreativität
generation_config = {
    "max_new_tokens": 1024,
    "temperature": 0.9,
    "top_p": 0.95,
    "do_sample": True
}
```

## Best Practices

1. **Klare Prompts**: Formulieren Sie Ihre Anfragen so präzise wie möglich
2. **Kontext bereitstellen**: Geben Sie ausreichend Hintergrundinformationen
3. **Experimentieren**: Testen Sie verschiedene Generierungsparameter
4. **Memory Management**: Geben Sie regelmäßig Speicher frei
5. **Batch-Verarbeitung**: Nutzen Sie Batches für multiple Anfragen
6. **Error Handling**: Implementieren Sie robuste Fehlerbehandlung

## Fehlerbehebung

### Häufige Probleme

**Leere Antworten:**
- Überprüfen Sie den `pad_token_id`
- Stellen Sie sicher, dass `add_generation_prompt=True` gesetzt ist

**Inkonsistente Ausgaben:**
- Reduzieren Sie `temperature` für deterministische Ergebnisse
- Verwenden Sie `do_sample=False` für greedy decoding

**Speicherprobleme:**
- Reduzieren Sie `max_new_tokens`
- Verwenden Sie kleinere Batch-Größen
- Aktivieren Sie Gradient Checkpointing
