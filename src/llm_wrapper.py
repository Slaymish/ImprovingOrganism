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
        self.device_info = self._get_device_info()
        
        if ML_AVAILABLE:
            try:
                logger.info(f"🤖 Found model: {settings.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
                
                # Check available GPU memory
                gpu_memory_available = False
                if torch and torch.cuda.is_available() and not settings.force_cpu:
                    try:
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                        logger.info(f"🖥️ GPU Memory: {gpu_memory:.2f}GB total")
                        
                        # Use GPU if we have enough memory and it's not forced to CPU
                        if gpu_memory > settings.max_gpu_memory_gb:
                            gpu_memory_available = True
                        else:
                            logger.warning(f"⚠️ Limited GPU memory ({gpu_memory:.2f}GB < {settings.max_gpu_memory_gb}GB), using CPU fallback")
                    except Exception as e:
                        logger.warning(f"Could not check GPU memory: {e}")
                elif settings.force_cpu:
                    logger.info("🖥️ CPU mode forced by configuration")
                
                # Configure model loading based on available memory
                if gpu_memory_available and settings.enable_memory_optimization:
                    # GPU loading with memory optimization
                    base_model = AutoModelForCausalLM.from_pretrained(
                        settings.model_name, 
                        torch_dtype=torch.float16,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        use_cache=False  # Disable cache to save memory
                    )
                    logger.info("🚀 Model loaded on GPU with memory optimization")
                else:
                    # CPU fallback for low memory systems
                    base_model = AutoModelForCausalLM.from_pretrained(
                        settings.model_name, 
                        torch_dtype=torch.float32,
                        device_map="cpu",
                        low_cpu_mem_usage=True
                    )
                    logger.info("💻 Model loaded on CPU (memory fallback)")
                
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

    def _get_device_info(self):
        """Get information about available compute devices"""
        device_info = {"type": "cpu", "memory_gb": 0, "available": True}
        
        if torch and torch.cuda.is_available():
            try:
                device_info["type"] = "cuda"
                device_info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                device_info["name"] = torch.cuda.get_device_name(0)
            except:
                pass
        
        return device_info

    def get_memory_status(self):
        """Get current memory usage information"""
        status = {"device": self.device_info["type"]}
        
        if torch and torch.cuda.is_available() and self.device_info["type"] == "cuda":
            try:
                status["allocated_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
                status["reserved_gb"] = torch.cuda.memory_reserved(0) / (1024**3)
                status["total_gb"] = self.device_info["memory_gb"]
                status["free_gb"] = status["total_gb"] - status["allocated_gb"]
                status["utilization"] = status["allocated_gb"] / status["total_gb"] * 100
            except:
                status["error"] = "Could not get GPU memory info"
        
        return status

    def clear_memory(self):
        """Clear GPU memory cache"""
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 GPU memory cache cleared")

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
            # Clear GPU cache before generation if available
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Format prompt for chat-style generation
            formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
            
            # Prepare input with shorter context for memory efficiency
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt",
                max_length=512,  # Limit input length to save memory
                truncation=True
            )
            
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Reduce max_tokens for memory-constrained systems
            effective_max_tokens = min(max_tokens, 100) if hasattr(self.model, 'device') and 'cuda' in str(self.model.device) else max_tokens
            
            # Generate output with memory-optimized parameters
            with torch.no_grad() if torch and hasattr(torch, 'no_grad') else nullcontext():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=effective_max_tokens,
                    min_new_tokens=10,  # Reduced minimum
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    use_cache=False,  # Disable caching to save memory
                    pad_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else 0,
                    eos_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else 0
                )
            
            # Clear cache again after generation
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Decode and extract only the new response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response (after the formatted prompt)
            if "<|assistant|>\n" in full_response:
                response = full_response.split("<|assistant|>\n")[1].strip()
            else:
                # Fallback: remove the original prompt
                response = full_response[len(formatted_prompt):].strip()
            
            return response if response else "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
        except torch.cuda.OutOfMemoryError if torch else Exception as e:
            # Handle CUDA OOM specifically
            logger.error(f"CUDA out of memory during generation. Falling back to shortened response.")
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Return a fallback response for CUDA OOM
            return f"I understand your question about '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'. Due to memory constraints, I'm providing a brief response. Please try a shorter prompt or restart the system for better performance."
            
        except Exception as e:
            logger.error(f"Error during text generation: {e}", exc_info=True)
            # Clear cache on any error
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return f"I encountered an error while generating a response. Please try again with a shorter prompt."

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
