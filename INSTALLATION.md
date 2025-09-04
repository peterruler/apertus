# Installationsanleitung

Diese Anleitung führt Sie durch die Installation und Einrichtung des Apertus-8B-Instruct-2509 Projekts.

## Systemanforderungen

### Minimal
- **Betriebssystem**: Windows 10/11, macOS 10.15+, oder Linux (Ubuntu 18.04+)
- **Python**: Version 3.8 bis 3.11
- **RAM**: Mindestens 16GB
- **Speicher**: 20GB freier Speicherplatz
- **Internet**: Für das Herunterladen des Modells (ca. 15GB)

### Empfohlen
- **RAM**: 32GB oder mehr
- **GPU**: NVIDIA GPU mit mindestens 8GB VRAM (RTX 3080, RTX 4080, RTX 4090, oder Tesla/Quadro Karten)
- **CUDA**: Version 11.8 oder 12.x
- **Speicher**: 50GB freier Speicherplatz

## Schritt-für-Schritt Installation

### 1. Repository klonen

```bash
git clone https://github.com/peterruler/apertus.git
cd apertus
```

### 2. Python Virtual Environment erstellen (empfohlen)

#### Für Linux/macOS:
```bash
python3 -m venv apertus-env
source apertus-env/bin/activate
```

#### Für Windows:
```cmd
python -m venv apertus-env
apertus-env\Scripts\activate
```

### 3. Abhängigkeiten installieren

#### CPU-Version (ohne GPU-Unterstützung):
```bash
pip install -r requirements.txt
```

#### GPU-Version (mit CUDA-Unterstützung):
```bash
# Für CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Für CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Dann die restlichen Abhängigkeiten
pip install transformers accelerate
```

### 4. Installation überprüfen

```bash
python -c "import torch; print(f'PyTorch Version: {torch.__version__}')"
python -c "import torch; print(f'CUDA verfügbar: {torch.cuda.is_available()}')"
python -c "from transformers import AutoTokenizer; print('Transformers erfolgreich installiert')"
```

## Plattform-spezifische Installationen

### macOS mit Apple Silicon (M1/M2/M3)

```bash
# Spezielle PyTorch-Version für Apple Silicon
pip install torch torchvision torchaudio
pip install transformers accelerate

# Optional: MPS-Unterstützung aktivieren
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Windows mit CUDA

1. **CUDA Toolkit installieren**:
   - Laden Sie CUDA 11.8 oder 12.1 von der NVIDIA-Website herunter
   - Installieren Sie das Toolkit mit Standardeinstellungen

2. **PyTorch mit CUDA installieren**:
   ```cmd
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Überprüfung**:
   ```cmd
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Linux (Ubuntu/Debian)

```bash
# System-Pakete aktualisieren
sudo apt update && sudo apt upgrade -y

# Python und pip sicherstellen
sudo apt install python3 python3-pip python3-venv

# NVIDIA-Treiber installieren (falls GPU vorhanden)
sudo apt install nvidia-driver-525  # oder neueste Version

# CUDA installieren (optional)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Dann normale Installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate
```

## Erste Verwendung

### 1. Schnelltest

```bash
python example_usage.py --mode single --prompt "Hallo, wie geht es dir?"
```

### 2. Interaktiver Chat

```bash
python example_usage.py --mode chat
```

### 3. Jupyter Notebook

```bash
jupyter notebook apertus.ipynb
```

## Fehlerbehebung

### Häufige Probleme und Lösungen

#### "CUDA out of memory"
```bash
# Verwenden Sie CPU statt GPU
python example_usage.py --device cpu

# Oder reduzieren Sie die Modellpräzision (in Python):
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Verwendet weniger VRAM
    device_map="auto"
)
```

#### "No module named 'torch'"
```bash
# Neuinstallation von PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### "transformers requires a Python version >= 3.8"
```bash
# Python-Version überprüfen
python --version

# Falls zu alt, Python aktualisieren oder pyenv verwenden
```

#### Modell lädt nicht
```bash
# Internet-Verbindung überprüfen
ping huggingface.co

# Manuell herunterladen (falls erforderlich)
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('swiss-ai/Apertus-8B-Instruct-2509')"
```

### Speicher-Optimierung

Für Systeme mit begrenztem RAM/VRAM:

```python
# In Python-Code:
import torch

# Gradient Checkpointing aktivieren
model.gradient_checkpointing_enable()

# Float16 verwenden
model = model.half()

# Speicher nach Generierung freigeben
def clear_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

## Performance-Optimierung

### Für maximale Geschwindigkeit:

```bash
# Installation mit Optimierungen
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers[torch] accelerate bitsandbytes
```

### Für minimalen Speicherverbrauch:

```bash
# Quantisierung aktivieren
pip install bitsandbytes

# In Python:
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # 8-bit Quantisierung
    device_map="auto"
)
```

## Nächste Schritte

Nach erfolgreicher Installation:

1. Lesen Sie die [Verwendungsanleitung](USAGE.md)
2. Schauen Sie sich die [Beispiele](EXAMPLES.md) an
3. Konsultieren Sie die [API-Referenz](API_REFERENCE.md)
4. Experimentieren Sie mit dem Jupyter Notebook

## Support

Bei Problemen:

1. Überprüfen Sie die [FAQ](README.md#fehlerbehebung)
2. Öffnen Sie ein Issue in diesem Repository
3. Konsultieren Sie die [Transformers-Dokumentation](https://huggingface.co/docs/transformers)
4. Besuchen Sie das [Hugging Face Forum](https://discuss.huggingface.co/)
