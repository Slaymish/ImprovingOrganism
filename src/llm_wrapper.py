import os
import logging
from typing import Optional, Union
from contextlib import nullcontext

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import guards for optional ML dependencies
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    torch = None
    
    # Define mock classes for when ML dependencies are not available
    class MockTokenizer:
        def __init__(self, *args, **kwargs): 
            self.eos_token_id = 0
        
        @classmethod
        def from_pretrained(cls, *args, **kwargs): 
            return cls()
        
        def __call__(self, *args, **kwargs): 
            return {'input_ids': [[1, 2, 3]], 'attention_mask': [[1, 1, 1]]}
        
        def decode(self, *args, **kwargs): 
            return "Mock response"

    class MockModel:
        def __init__(self, *args, **kwargs): 
            self.device = "cpu"
        
        @classmethod
        def from_pretrained(cls, *args, **kwargs): 
            return cls()
        
        def generate(self, *args, **kwargs): 
            return [[1, 2, 3, 4, 5]]

    AutoTokenizer = MockTokenizer
    AutoModelForCausalLM = MockModel
    PeftModel = MockModel

from .config import settings

class LLMWrapper:
    def __init__(self):
        self.model: Optional[PeftModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        
        if ML_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
                base_model = AutoModelForCausalLM.from_pretrained(
                    settings.model_name, 
                    torch_dtype=torch.float16 if torch else None, 
                    device_map="auto"
                )
                
                # Load LoRA adapter if it exists
                if os.path.exists(settings.lora_path):
                    logger.info(f"✅ Found LoRA adapter at {settings.lora_path}, loading...")
                    self.model = PeftModel.from_pretrained(base_model, settings.lora_path)
                else:
                    logger.warning(f"⚠️ LoRA adapter not found at {settings.lora_path}. Using base model only.")
                    self.model = base_model
                    
            except Exception as e:
                logger.error(f"Failed to load ML model: {e}", exc_info=True)
                # Fall back to mock mode
                self.tokenizer = AutoTokenizer()
                self.model = AutoModelForCausalLM()
        else:
            logger.warning("ML dependencies not available. Running in mock mode.")
            self.tokenizer = AutoTokenizer()
            self.model = AutoModelForCausalLM()

    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        if not ML_AVAILABLE or self.model is None or self.tokenizer is None:
            # Mock generation for development
            mock_responses = {
                "explain machine learning": "Machine learning is a branch of artificial intelligence where computers learn patterns from data to make predictions or decisions without being explicitly programmed for each task.",
                "invent something": "I'd like to invent a 'Memory Garden' - a digital space where people can plant virtual seeds representing memories, watch them grow into stories, and share them with loved ones across generations.",
                "neural networks": "Neural networks are computer systems inspired by how the human brain works. They consist of interconnected nodes (like brain neurons) that process information and learn patterns from examples.",
                "data privacy": "Data privacy is crucial because it protects personal information from misuse, maintains trust in digital services, prevents identity theft, and preserves individual autonomy in an increasingly connected world."
            }
            
            # Find the best matching response
            prompt_lower = prompt.lower()
            for key, response in mock_responses.items():
                if key in prompt_lower:
                    return response
            
            return f"I understand you're asking about '{prompt}'. This is a mock response while the full AI model loads. The system is designed to provide thoughtful, helpful answers to your questions."
            
        try:
            # Format prompt for chat-style generation
            formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
            
            # Prepare input
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate output with better parameters
            with torch.no_grad() if torch and hasattr(torch, 'no_grad') else nullcontext():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    min_new_tokens=20,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else 0,
                    eos_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else 0
                )
            
            # Decode and extract only the new response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response (after the formatted prompt)
            if "<|assistant|>\n" in full_response:
                response = full_response.split("<|assistant|>\n")[1].strip()
            else:
                # Fallback: remove the original prompt
                response = full_response[len(formatted_prompt):].strip()
            
            return response if response else "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
        except Exception as e:
            logger.error(f"Error during text generation: {e}", exc_info=True)
            return f"I encountered an error while generating a response: {str(e)}. Please try again."

    def get_embeddings(self, text: str):
        """Get embeddings for the given text"""
        if not ML_AVAILABLE or self.model is None or self.tokenizer is None:
            # Mock embeddings for development
            logger.info("Generating mock embeddings.")
            import numpy as np
            return np.random.rand(768)  # Common embedding size
            
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            if torch and hasattr(torch, 'no_grad'):
                with torch.no_grad():
                    outputs = self.model.base_model(**inputs, output_hidden_states=True)
            else:
                outputs = self.model.base_model(**inputs, output_hidden_states=True)
            
            # Use the last hidden state as the embedding
            embeddings = outputs.hidden_states[-1].mean(dim=1).squeeze()
            return embeddings.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}", exc_info=True)
            import numpy as np
            return np.random.rand(768)  # Fallback to mock
            return None
