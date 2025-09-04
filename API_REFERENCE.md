# API Referenz

Diese Dokumentation beschreibt die wichtigsten Klassen und Methoden zur Nutzung des Apertus-8B-Instruct-2509 Modells.

## AutoTokenizer

### Initialisierung

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("swiss-ai/Apertus-8B-Instruct-2509")
```

### Wichtige Methoden

#### `apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`

Formatiert Nachrichten für das Chat-Format.

**Parameter:**
- `messages` (List[Dict]): Liste von Nachrichten im Format `{"role": "user|assistant", "content": "text"}`
- `tokenize` (bool): Ob der Text tokenisiert werden soll (Standard: False)
- `add_generation_prompt` (bool): Ob ein Generierungs-Prompt hinzugefügt werden soll

**Rückgabe:**
- `str`: Formatierter Text

**Beispiel:**
```python
messages = [{"role": "user", "content": "Hallo!"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

#### `encode(text, return_tensors="pt")`

Tokenisiert Text zu Token-IDs.

**Parameter:**
- `text` (str): Eingabetext
- `return_tensors` (str): Format der Rückgabe ("pt" für PyTorch)

**Beispiel:**
```python
tokens = tokenizer.encode("Hallo Welt", return_tensors="pt")
```

#### `decode(token_ids, skip_special_tokens=True)`

Dekodiert Token-IDs zurück zu Text.

**Parameter:**
- `token_ids` (Tensor): Token-IDs
- `skip_special_tokens` (bool): Spezielle Token überspringen

**Beispiel:**
```python
text = tokenizer.decode(token_ids, skip_special_tokens=True)
```

## AutoModelForCausalLM

### Initialisierung

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "swiss-ai/Apertus-8B-Instruct-2509",
    torch_dtype=torch.float16,  # Optional: für Memory-Effizienz
    device_map="auto"           # Optional: automatische Geräte-Zuordnung
)
```

### Wichtige Methoden

#### `generate(**kwargs)`

Generiert Text basierend auf Eingabe.

**Wichtige Parameter:**
- `input_ids` (Tensor): Eingabe-Token-IDs
- `max_new_tokens` (int): Maximale Anzahl neuer Token (Standard: 20)
- `temperature` (float): Sampling-Temperatur (0.0-2.0, Standard: 1.0)
- `top_p` (float): Nucleus sampling Parameter (0.0-1.0, Standard: 1.0)
- `top_k` (int): Top-K sampling Parameter
- `do_sample` (bool): Aktiviert Sampling (Standard: False)
- `repetition_penalty` (float): Wiederholungsstrafe (Standard: 1.0)
- `pad_token_id` (int): Padding Token ID

**Rückgabe:**
- `torch.Tensor`: Generierte Token-IDs

**Beispiel:**
```python
generated_ids = model.generate(
    input_ids=model_inputs.input_ids,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)
```

#### `forward(input_ids, attention_mask=None, **kwargs)`

Führt einen Forward-Pass durch das Modell aus.

**Parameter:**
- `input_ids` (Tensor): Eingabe-Token-IDs
- `attention_mask` (Tensor): Attention-Maske

## Generierungsparameter

### Temperatur (temperature)

Kontrolliert die Zufälligkeit der Ausgabe:
- `0.0`: Deterministisch (greedy)
- `0.1-0.5`: Konservativ
- `0.6-0.8`: Ausgewogen
- `0.9-1.2`: Kreativ
- `>1.2`: Sehr zufällig

```python
# Deterministisch
generated_ids = model.generate(input_ids, temperature=0.0, do_sample=False)

# Kreativ
generated_ids = model.generate(input_ids, temperature=0.9, do_sample=True)
```

### Top-p (Nucleus Sampling)

Begrenzt die Sampling-Auswahl auf die wahrscheinlichsten Token:
- `0.1`: Sehr konservativ
- `0.5`: Mäßig konservativ
- `0.9`: Standard
- `1.0`: Alle Token berücksichtigen

```python
generated_ids = model.generate(
    input_ids,
    top_p=0.9,
    do_sample=True
)
```

### Top-k Sampling

Begrenzt die Auswahl auf die k wahrscheinlichsten Token:

```python
generated_ids = model.generate(
    input_ids,
    top_k=50,
    do_sample=True
)
```

### Repetition Penalty

Verhindert Wiederholungen:
- `1.0`: Keine Strafe
- `1.1-1.2`: Leichte Strafe
- `1.3-1.5`: Starke Strafe

```python
generated_ids = model.generate(
    input_ids,
    repetition_penalty=1.1
)
```

## Utility-Funktionen

### Memory Management

```python
import torch
import gc

def clear_gpu_memory():
    """Räumt GPU-Speicher auf."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_gpu_memory_usage():
    """Zeigt GPU-Speicherverbrauch an."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
    return "CUDA nicht verfügbar"
```

### Text Processing

```python
def prepare_chat_input(user_message, conversation_history=None):
    """Bereitet Chat-Eingabe vor."""
    if conversation_history is None:
        conversation_history = []
    
    messages = conversation_history + [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def extract_response(generated_ids, input_length):
    """Extrahiert die Antwort aus generierten Token-IDs."""
    output_ids = generated_ids[0][input_length:]
    return tokenizer.decode(output_ids, skip_special_tokens=True)
```

### Batch Processing

```python
def batch_generate(prompts, batch_size=4, **generation_kwargs):
    """Generiert Antworten für mehrere Prompts in Batches."""
    responses = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Prompts vorbereiten
        messages_batch = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]
        texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
                for msg in messages_batch]
        
        # Tokenisieren
        model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        
        # Generieren
        with torch.no_grad():
            generated_ids = model.generate(**model_inputs, **generation_kwargs)
        
        # Dekodieren
        for j, generated in enumerate(generated_ids):
            output_ids = generated[len(model_inputs.input_ids[j]):]
            response = tokenizer.decode(output_ids, skip_special_tokens=True)
            responses.append(response)
        
        clear_gpu_memory()
    
    return responses
```

## Fehlerbehandlung

### Typische Exceptions

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    model = AutoModelForCausalLM.from_pretrained("swiss-ai/Apertus-8B-Instruct-2509")
except Exception as e:
    if "out of memory" in str(e).lower():
        print("GPU-Speicher nicht ausreichend. Versuche CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            "swiss-ai/Apertus-8B-Instruct-2509",
            device_map="cpu"
        )
    else:
        raise e

def safe_generate(model, tokenizer, prompt, max_retries=3):
    """Sichere Generierung mit Wiederholungsversuchen."""
    for attempt in range(max_retries):
        try:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            generated_ids = model.generate(**model_inputs, max_new_tokens=512)
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
            response = tokenizer.decode(output_ids, skip_special_tokens=True)
            
            return response
            
        except torch.cuda.OutOfMemoryError:
            clear_gpu_memory()
            if attempt == max_retries - 1:
                raise
            print(f"OOM Fehler, Versuch {attempt + 1} von {max_retries}")
            
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Fehler: {e}, Versuch {attempt + 1} von {max_retries}")
    
    return None
```

## Performance Monitoring

```python
import time
import psutil

def benchmark_generation(model, tokenizer, prompt, num_runs=5):
    """Benchmarkt die Generierungsleistung."""
    times = []
    
    for i in range(num_runs):
        start_time = time.time()
        
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(**model_inputs, max_new_tokens=256)
        
        end_time = time.time()
        times.append(end_time - start_time)
        
        clear_gpu_memory()
    
    avg_time = sum(times) / len(times)
    print(f"Durchschnittliche Generierungszeit: {avg_time:.2f}s")
    print(f"Token pro Sekunde: {256 / avg_time:.2f}")
    
    return times
```
