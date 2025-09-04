# Beispiele

Diese Sammlung zeigt verschiedene Anwendungsfälle für das Apertus-8B-Instruct-2509 Modell.

## Grundlegende Beispiele

### 1. Einfache Frage-Antwort

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Modell laden
model_name = "swiss-ai/Apertus-8B-Instruct-2509"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

# Einfache Frage
prompt = "Was ist der Unterschied zwischen KI und maschinellem Lernen?"
messages = [{"role": "user", "content": prompt}]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

generated_ids = model.generate(**model_inputs, max_new_tokens=512, temperature=0.7)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
response = tokenizer.decode(output_ids, skip_special_tokens=True)

print(response)
```

### 2. Kreatives Schreiben

```python
prompt = """
Schreibe eine kurze Geschichte (200 Wörter) über einen Astronauten, 
der auf einem fremden Planeten eine mysteriöse Nachricht findet.
"""

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

# Höhere Temperatur für Kreativität
generated_ids = model.generate(
    **model_inputs, 
    max_new_tokens=300,
    temperature=0.9,
    top_p=0.95,
    do_sample=True
)

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
story = tokenizer.decode(output_ids, skip_special_tokens=True)
print(story)
```

## Code-Beispiele

### 3. Python Code-Generierung

```python
prompt = """
Erstelle eine Python-Klasse für einen Taschenrechner mit folgenden Methoden:
- Addition
- Subtraktion  
- Multiplikation
- Division (mit Fehlerbehandlung)
- Potenzierung

Füge Docstrings und Beispiele hinzu.
"""

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=800,
    temperature=0.3,  # Niedrigere Temperatur für präzisen Code
    do_sample=True
)

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
code = tokenizer.decode(output_ids, skip_special_tokens=True)
print(code)
```

### 4. Code-Erklärung

```python
code_snippet = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""

prompt = f"""
Erkläre diesen Python-Code Schritt für Schritt:

{code_snippet}

Gehe auf die Algorithmus-Komplexität ein und erkläre, warum dieser Ansatz effizient ist.
"""

messages = [{"role": "user", "content": prompt}]
# ... Generierung wie oben
```

## Konversations-Beispiele

### 5. Multi-Turn Dialog

```python
def chat_conversation():
    conversation = []
    
    while True:
        user_input = input("Sie: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            break
            
        # Nachricht zur Konversation hinzufügen
        conversation.append({"role": "user", "content": user_input})
        
        # Text formatieren
        text = tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
        
        # Antwort generieren
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        print(f"Apertus: {response}")
        
        # Antwort zur Konversation hinzufügen
        conversation.append({"role": "assistant", "content": response})

# chat_conversation()
```

### 6. Rollenspiel-Chat

```python
system_prompt = """
Du bist ein erfahrener Koch in einem französischen Restaurant. 
Antworte enthusiastisch und teile dein Wissen über französische Küche.
Verwende gelegentlich französische Begriffe und erkläre sie.
"""

conversation = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Wie bereite ich ein perfektes Coq au Vin zu?"}
]

text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=400,
    temperature=0.8,
    do_sample=True
)

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
response = tokenizer.decode(output_ids, skip_special_tokens=True)
print(response)
```

## Textanalyse-Beispiele

### 7. Sentiment-Analyse

```python
text_to_analyze = """
Das neue Smartphone ist wirklich beeindruckend! Die Kamera macht fantastische Fotos 
und die Akkulaufzeit ist deutlich besser als beim Vorgängermodell. Allerdings ist 
der Preis ziemlich hoch und die Bedienung könnte intuitiver sein.
"""

prompt = f"""
Analysiere das Sentiment des folgenden Textes und identifiziere:
1. Gesamtstimmung (positiv/negativ/neutral)
2. Spezifische positive Aspekte
3. Spezifische negative Aspekte
4. Bewertung auf einer Skala von 1-10

Text: {text_to_analyze}
"""

messages = [{"role": "user", "content": prompt}]
# ... Generierung
```

### 8. Textzusammenfassung

```python
long_text = """
Der Klimawandel ist eine der größten Herausforderungen unserer Zeit. Die globale 
Durchschnittstemperatur ist in den letzten 100 Jahren um etwa 1,1 Grad Celsius 
gestiegen. Hauptursache sind Treibhausgase wie CO2, die durch menschliche Aktivitäten 
freigesetzt werden, insbesondere durch die Verbrennung fossiler Brennstoffe.

Die Auswirkungen sind bereits spürbar: Gletscher schmelzen, der Meeresspiegel steigt, 
Extremwetterereignisse nehmen zu. Ohne drastische Maßnahmen zur Reduktion der 
Treibhausgasemissionen wird sich die Situation weiter verschärfen.

Lösungsansätze umfassen den Ausbau erneuerbarer Energien, Energieeffizienz, 
nachhaltige Mobilität und Änderungen im Konsumverhalten. Internationale Zusammenarbeit 
ist entscheidend, wie das Pariser Klimaabkommen zeigt.
"""

prompt = f"""
Fasse den folgenden Text in 3-4 Sätzen zusammen und extrahiere die wichtigsten Punkte:

{long_text}
"""

messages = [{"role": "user", "content": prompt}]
# ... Generierung
```

## Spezielle Anwendungen

### 9. Übersetzung

```python
prompt = """
Übersetze den folgenden deutschen Text ins Englische und erkläre 
schwierige Begriffe oder kulturelle Besonderheiten:

"Das Oktoberfest in München ist mehr als nur ein Volksfest. Es ist ein 
wichtiger Wirtschaftsfaktor und zieht jährlich Millionen von Besuchern 
aus aller Welt an, die das bayerische Brauchtum erleben möchten."
"""

messages = [{"role": "user", "content": prompt}]
# ... Generierung
```

### 10. Mathe-Tutoring

```python
prompt = """
Erkläre Schritt für Schritt, wie man diese Gleichung löst:
2x² + 5x - 3 = 0

Verwende die quadratische Formel und erkläre jeden Schritt ausführlich.
"""

messages = [{"role": "user", "content": prompt}]

# Präzise Antwort für Mathe
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=600,
    temperature=0.1,  # Sehr niedrig für Genauigkeit
    do_sample=True
)
```

## Batch-Verarbeitung

### 11. Multiple Prompts verarbeiten

```python
prompts = [
    "Erkläre Photosynthese in einfachen Worten.",
    "Was ist der Unterschied zwischen Virus und Bakterie?",
    "Wie funktioniert ein Computer-Prozessor?",
    "Beschreibe den Wasserkreislauf."
]

def batch_process(prompts, batch_size=2):
    all_responses = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        
        # Batch vorbereiten
        messages_batch = [[{"role": "user", "content": prompt}] for prompt in batch]
        texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
                for msg in messages_batch]
        
        # Tokenisieren mit Padding
        model_inputs = tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to("cuda")
        
        # Generieren
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Dekodieren
        for j, generated in enumerate(generated_ids):
            input_length = len(model_inputs.input_ids[j])
            output_ids = generated[input_length:]
            response = tokenizer.decode(output_ids, skip_special_tokens=True)
            all_responses.append(response)
    
    return all_responses

responses = batch_process(prompts)
for prompt, response in zip(prompts, responses):
    print(f"Frage: {prompt}")
    print(f"Antwort: {response}\n")
```

### 12. Streaming-Generierung (simuliert)

```python
def streaming_generate(prompt, chunk_size=50):
    """Simuliert Streaming-Generierung durch schrittweise Dekodierung."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    # Vollständige Generierung
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=500,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Ausgabe-Token extrahieren
    input_length = len(model_inputs.input_ids[0])
    output_ids = generated_ids[0][input_length:]
    
    # Schrittweise dekodieren
    full_response = ""
    for i in range(0, len(output_ids), chunk_size):
        chunk_ids = output_ids[:i + chunk_size]
        current_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        
        # Nur neuen Teil ausgeben
        new_text = current_text[len(full_response):]
        print(new_text, end='', flush=True)
        full_response = current_text
        
        import time
        time.sleep(0.1)  # Simuliert Streaming-Delay
    
    print()  # Neue Zeile am Ende
    return full_response

# Verwendung
response = streaming_generate("Erkläre mir die Evolutionstheorie.")
```

## Performance-Tests

### 13. Benchmark verschiedener Parameter

```python
import time

def benchmark_parameters():
    prompt = "Schreibe einen kurzen Essay über künstliche Intelligenz."
    results = []
    
    parameter_sets = [
        {"temperature": 0.1, "top_p": 0.9, "max_new_tokens": 256},
        {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 256},
        {"temperature": 0.9, "top_p": 0.95, "max_new_tokens": 256},
    ]
    
    for i, params in enumerate(parameter_sets):
        print(f"Test {i+1}: {params}")
        
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
        
        start_time = time.time()
        generated_ids = model.generate(
            **model_inputs,
            **params,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        end_time = time.time()
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        results.append({
            "parameters": params,
            "time": end_time - start_time,
            "response_length": len(response),
            "response": response[:100] + "..."
        })
    
    return results

# results = benchmark_parameters()
```

Diese Beispiele zeigen die Vielseitigkeit des Apertus-8B-Instruct-2509 Modells und bieten praktische Anwendungsfälle für verschiedene Szenarien.
