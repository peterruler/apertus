#!/usr/bin/env python3
"""
Beispiel-Script für die Verwendung des Apertus-8B-Instruct-2509 Modells.

Dieses Script demonstriert grundlegende Funktionen wie:
- Modell laden
- Einfache Textgenerierung  
- Chat-Konversation
- Batch-Verarbeitung
"""

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import time


class ApertusChat:
    """Wrapper-Klasse für das Apertus-8B-Instruct-2509 Modell."""
    
    def __init__(self, device="auto", model_name="swiss-ai/Apertus-8B-Instruct-2509"):
        """
        Initialisiert das Modell.
        
        Args:
            device (str): Gerät für das Modell ("cuda", "cpu", "auto")
            model_name (str): Name des Hugging Face Modells
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        
        print(f"Lade Modell auf {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Modell-Konfiguration je nach Gerät
        model_kwargs = {}
        if self.device == "cuda":
            model_kwargs.update({
                "torch_dtype": torch.float16,
                "device_map": "auto"
            })
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        ).to(self.device)
        
        print("Modell erfolgreich geladen!")
    
    def _setup_device(self, device):
        """Bestimmt das optimale Gerät."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def generate_response(self, prompt, max_new_tokens=512, temperature=0.7, 
                         top_p=0.9, do_sample=True):
        """
        Generiert eine Antwort auf einen Prompt.
        
        Args:
            prompt (str): Eingabe-Prompt
            max_new_tokens (int): Maximale Anzahl neuer Token
            temperature (float): Sampling-Temperatur
            top_p (float): Nucleus sampling Parameter
            do_sample (bool): Aktiviert Sampling
            
        Returns:
            str: Generierte Antwort
        """
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Speicher freigeben
        self._clear_memory()
        
        return response
    
    def chat_conversation(self):
        """Startet eine interaktive Chat-Konversation."""
        print("\n=== Apertus Chat ===")
        print("Geben Sie 'quit', 'exit' oder 'bye' ein, um den Chat zu beenden.")
        print("-" * 50)
        
        conversation = []
        
        while True:
            try:
                user_input = input("\nSie: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("Auf Wiedersehen!")
                    break
                
                if not user_input:
                    continue
                
                # Nachricht zur Konversation hinzufügen
                conversation.append({"role": "user", "content": user_input})
                
                # Antwort generieren
                text = self.tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
                
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
                
                print("Apertus: ", end="", flush=True)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=256,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
                response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                
                print(response)
                
                # Antwort zur Konversation hinzufügen
                conversation.append({"role": "assistant", "content": response})
                
                self._clear_memory()
                
            except KeyboardInterrupt:
                print("\n\nChat beendet.")
                break
            except Exception as e:
                print(f"\nFehler: {e}")
                continue
    
    def batch_generate(self, prompts, batch_size=2, **generation_kwargs):
        """
        Generiert Antworten für mehrere Prompts in Batches.
        
        Args:
            prompts (List[str]): Liste von Prompts
            batch_size (int): Batch-Größe
            **generation_kwargs: Parameter für generate()
            
        Returns:
            List[str]: Liste von Antworten
        """
        responses = []
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        
        for i in range(0, len(prompts), batch_size):
            batch_num = i // batch_size + 1
            print(f"Verarbeite Batch {batch_num}/{total_batches}...")
            
            batch_prompts = prompts[i:i + batch_size]
            
            # Prompts vorbereiten
            messages_batch = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]
            texts = [self.tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            ) for msg in messages_batch]
            
            # Tokenisieren mit Padding
            model_inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            
            # Generieren
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=generation_kwargs.get('max_new_tokens', 256),
                    temperature=generation_kwargs.get('temperature', 0.7),
                    top_p=generation_kwargs.get('top_p', 0.9),
                    do_sample=generation_kwargs.get('do_sample', True),
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Dekodieren
            for j, generated in enumerate(generated_ids):
                input_length = len(model_inputs.input_ids[j])
                output_ids = generated[input_length:]
                response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                responses.append(response)
            
            self._clear_memory()
        
        return responses
    
    def _clear_memory(self):
        """Räumt Speicher auf."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model_info(self):
        """Gibt Informationen über das Modell aus."""
        info = {
            "Modell": self.model_name,
            "Gerät": self.device,
            "Parameter": f"{self.model.num_parameters():,}",
        }
        
        if torch.cuda.is_available():
            info["GPU"] = torch.cuda.get_device_name()
            info["VRAM verfügbar"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        
        return info


def main():
    parser = argparse.ArgumentParser(description="Apertus-8B-Instruct-2509 Demo")
    parser.add_argument("--device", default="auto", choices=["cuda", "cpu", "auto"],
                       help="Gerät für das Modell")
    parser.add_argument("--mode", default="chat", choices=["chat", "single", "batch"],
                       help="Ausführungsmodus")
    parser.add_argument("--prompt", default="Hallo! Wie geht es dir?",
                       help="Prompt für single mode")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling-Temperatur")
    
    args = parser.parse_args()
    
    try:
        # Modell initialisieren
        chat = ApertusChat(device=args.device)
        
        # Modell-Informationen anzeigen
        print("\n=== Modell-Informationen ===")
        for key, value in chat.get_model_info().items():
            print(f"{key}: {value}")
        
        if args.mode == "chat":
            # Interaktiver Chat
            chat.chat_conversation()
            
        elif args.mode == "single":
            # Einzelner Prompt
            print(f"\nPrompt: {args.prompt}")
            print("-" * 50)
            
            start_time = time.time()
            response = chat.generate_response(
                args.prompt, 
                temperature=args.temperature
            )
            end_time = time.time()
            
            print(f"Antwort: {response}")
            print(f"\nGenerierungszeit: {end_time - start_time:.2f}s")
            
        elif args.mode == "batch":
            # Batch-Verarbeitung Demo
            demo_prompts = [
                "Erkläre mir die Relativitätstheorie in einfachen Worten.",
                "Was ist der Unterschied zwischen KI und maschinellem Lernen?",
                "Schreibe ein kurzes Gedicht über den Frühling.",
                "Wie bereite ich eine perfekte Pizza zu?"
            ]
            
            print(f"\nVerarbeite {len(demo_prompts)} Prompts in Batches...")
            print("-" * 50)
            
            start_time = time.time()
            responses = chat.batch_generate(demo_prompts, batch_size=2)
            end_time = time.time()
            
            for i, (prompt, response) in enumerate(zip(demo_prompts, responses), 1):
                print(f"\n{i}. Prompt: {prompt}")
                print(f"   Antwort: {response[:100]}...")
            
            print(f"\nGesamtzeit: {end_time - start_time:.2f}s")
            print(f"Durchschnitt pro Prompt: {(end_time - start_time) / len(demo_prompts):.2f}s")
            
    except KeyboardInterrupt:
        print("\nProgramm beendet.")
    except Exception as e:
        print(f"Fehler: {e}")


if __name__ == "__main__":
    main()
