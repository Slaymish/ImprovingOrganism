try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    
from .config import settings

class LLMWrapper:
    def __init__(self):
        if not ML_AVAILABLE:
            print("⚠️  ML dependencies not available. Using mock LLM for development.")
            self.tokenizer = None
            self.model = None
            return
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
            base_model = AutoModelForCausalLM.from_pretrained(settings.model_name, torch_dtype=torch.float16, device_map="auto")
            self.model = PeftModel.from_pretrained(base_model, settings.lora_path)
        except Exception as e:
            print(f"⚠️  Failed to load ML model: {e}. Using mock mode.")
            self.tokenizer = None
            self.model = None

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        if not ML_AVAILABLE or self.model is None:
            # Mock response for development
            return f"Mock response to: '{prompt[:50]}...' (This is a development placeholder. Install ML dependencies for real responses.)"
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
