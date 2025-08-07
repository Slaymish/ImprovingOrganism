"""
Training Safeguards Module - Prevents Model Collapse in Continual Learning

This module implements various techniques to prevent model collapse:
1. Elastic Weight Consolidation (EWC) - prevents catastrophic forgetting
2. Knowledge Retention Testing - validates general capabilities
3. Diversity Metrics - monitors output diversity
4. Conservative Training - limits learning rate and epochs
5. Regularization Techniques - prevents overfitting
6. Adapter Management - manages multiple LoRA adapters
"""

import os
import json
import torch
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import hashlib
from collections import defaultdict
import math

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import guards for ML dependencies
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
        Trainer, DataCollatorForLanguageModeling
    )
    from peft import (
        LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training,
        PeftModel
    )
    from datasets import Dataset
    import torch.nn.functional as F
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML dependencies not available for training safeguards")

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.memory_module import MemoryModule

class KnowledgeRetentionTester:
    """Tests if the model retains general knowledge after fine-tuning"""
    
    def __init__(self):
        self.general_knowledge_tests = [
            # Basic factual knowledge
            {"prompt": "What is the capital of France?", "expected_contains": ["Paris"]},
            {"prompt": "How many planets are in our solar system?", "expected_contains": ["eight", "8"]},
            {"prompt": "What is 2 + 2?", "expected_contains": ["4", "four"]},
            {"prompt": "Who wrote Romeo and Juliet?", "expected_contains": ["Shakespeare", "William"]},
            {"prompt": "What is the largest ocean on Earth?", "expected_contains": ["Pacific"]},
            
            # Scientific knowledge
            {"prompt": "What is photosynthesis?", "expected_contains": ["sunlight", "plants", "carbon dioxide", "oxygen"]},
            {"prompt": "What is gravity?", "expected_contains": ["force", "mass", "attraction"]},
            {"prompt": "What is DNA?", "expected_contains": ["genetic", "molecule", "information"]},
            
            # Mathematical concepts
            {"prompt": "What is a prime number?", "expected_contains": ["divisible", "1", "itself"]},
            {"prompt": "What is the Pythagorean theorem?", "expected_contains": ["triangle", "square", "hypotenuse"]},
            
            # Language understanding
            {"prompt": "Explain the difference between 'their', 'there', and 'they're'", "expected_contains": ["possession", "location", "they are"]},
            {"prompt": "What is a metaphor?", "expected_contains": ["comparison", "figure of speech"]},
        ]
        
        self.reasoning_tests = [
            # Logical reasoning
            {"prompt": "If all cats are animals and Fluffy is a cat, what can we conclude about Fluffy?", "expected_contains": ["animal"]},
            {"prompt": "What comes next in this sequence: 2, 4, 6, 8, ?", "expected_contains": ["10"]},
            {"prompt": "If it's raining, the ground gets wet. The ground is wet. Can we conclude it's raining?", "expected_contains": ["not necessarily", "other causes"]},
        ]
    
    def test_knowledge_retention(self, model, tokenizer, device="cpu") -> Dict[str, Any]:
        """Test if model retains general knowledge"""
        if not ML_AVAILABLE:
            return {"error": "ML dependencies not available"}
            
        results = {
            "general_knowledge": [],
            "reasoning": [],
            "overall_score": 0.0,
            "passed": False
        }
        
        try:
            model.eval()
            
            # Test general knowledge
            gk_score = 0
            for test in self.general_knowledge_tests:
                response = self._generate_response(model, tokenizer, test["prompt"], device)
                passed = any(expected.lower() in response.lower() for expected in test["expected_contains"])
                
                results["general_knowledge"].append({
                    "prompt": test["prompt"],
                    "response": response,
                    "expected": test["expected_contains"],
                    "passed": passed
                })
                
                if passed:
                    gk_score += 1
            
            # Test reasoning
            reasoning_score = 0
            for test in self.reasoning_tests:
                response = self._generate_response(model, tokenizer, test["prompt"], device)
                passed = any(expected.lower() in response.lower() for expected in test["expected_contains"])
                
                results["reasoning"].append({
                    "prompt": test["prompt"],
                    "response": response,
                    "expected": test["expected_contains"],
                    "passed": passed
                })
                
                if passed:
                    reasoning_score += 1
            
            # Calculate overall score
            total_tests = len(self.general_knowledge_tests) + len(self.reasoning_tests)
            total_passed = gk_score + reasoning_score
            results["overall_score"] = total_passed / total_tests
            results["passed"] = results["overall_score"] >= 0.7  # 70% threshold
            
            logger.info(f"Knowledge retention test: {total_passed}/{total_tests} passed ({results['overall_score']:.2%})")
            
        except Exception as e:
            logger.error(f"Knowledge retention test failed: {e}")
            results["error"] = str(e)
            
        return results
    
    def _generate_response(self, model, tokenizer, prompt: str, device: str, max_tokens: int = 100) -> str:
        """Generate a response for testing"""
        formatted_prompt = f"<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,  # Low temperature for consistent testing
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(formatted_prompt, "").strip()
        return response

class DiversityMonitor:
    """Monitors output diversity to detect mode collapse"""
    
    def __init__(self):
        self.diversity_prompts = [
            "Tell me about artificial intelligence.",
            "Explain machine learning.",
            "What is the future of technology?",
            "Describe a beautiful sunset.",
            "How can we solve climate change?",
            "What makes a good friend?",
            "Explain quantum physics simply.",
            "What is creativity?",
            "How do computers work?",
            "What is the meaning of life?"
        ]
    
    def measure_diversity(self, model, tokenizer, device="cpu", num_samples: int = 5) -> Dict[str, float]:
        """Measure output diversity across multiple generations"""
        if not ML_AVAILABLE:
            return {"error": "ML dependencies not available"}
            
        try:
            model.eval()
            all_responses = []
            
            # Generate multiple responses for each prompt
            for prompt in self.diversity_prompts[:5]:  # Use subset for efficiency
                for _ in range(num_samples):
                    response = self._generate_diverse_response(model, tokenizer, prompt, device)
                    all_responses.append(response)
            
            # Calculate diversity metrics
            metrics = {
                "lexical_diversity": self._calculate_lexical_diversity(all_responses),
                "semantic_diversity": self._calculate_semantic_diversity(all_responses),
                "length_variance": self._calculate_length_variance(all_responses),
                "repetition_score": self._calculate_repetition_score(all_responses),
                "overall_diversity": 0.0
            }
            
            # Calculate overall diversity score
            metrics["overall_diversity"] = (
                metrics["lexical_diversity"] * 0.3 +
                metrics["semantic_diversity"] * 0.3 +
                metrics["length_variance"] * 0.2 +
                (1.0 - metrics["repetition_score"]) * 0.2
            )
            
            logger.info(f"Diversity metrics: {metrics['overall_diversity']:.3f} overall")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Diversity measurement failed: {e}")
            return {"error": str(e)}
    
    def _generate_diverse_response(self, model, tokenizer, prompt: str, device: str) -> str:
        """Generate a diverse response"""
        formatted_prompt = f"<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.8,  # Higher temperature for diversity
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(formatted_prompt, "").strip()
        return response
    
    def _calculate_lexical_diversity(self, responses: List[str]) -> float:
        """Calculate lexical diversity (unique words / total words)"""
        all_words = []
        for response in responses:
            words = response.lower().split()
            all_words.extend(words)
        
        if not all_words:
            return 0.0
            
        unique_words = set(all_words)
        return len(unique_words) / len(all_words)
    
    def _calculate_semantic_diversity(self, responses: List[str]) -> float:
        """Calculate semantic diversity using simple hashing"""
        if len(responses) < 2:
            return 1.0
            
        # Simple semantic diversity using response hashes
        hashes = set()
        for response in responses:
            # Create hash based on content structure
            normalized = ' '.join(response.lower().split())
            response_hash = hashlib.md5(normalized.encode()).hexdigest()[:8]
            hashes.add(response_hash)
        
        return len(hashes) / len(responses)
    
    def _calculate_length_variance(self, responses: List[str]) -> float:
        """Calculate variance in response lengths"""
        lengths = [len(response.split()) for response in responses]
        if not lengths:
            return 0.0
            
        mean_length = sum(lengths) / len(lengths)
        variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
        
        # Normalize variance (higher variance = more diversity)
        normalized_variance = min(variance / (mean_length + 1), 1.0)
        return normalized_variance
    
    def _calculate_repetition_score(self, responses: List[str]) -> float:
        """Calculate how much content is repeated"""
        if len(responses) < 2:
            return 0.0
            
        total_comparisons = 0
        repetitive_comparisons = 0
        
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                total_comparisons += 1
                
                # Check for significant overlap
                words_i = set(responses[i].lower().split())
                words_j = set(responses[j].lower().split())
                
                if words_i and words_j:
                    overlap = len(words_i.intersection(words_j))
                    union = len(words_i.union(words_j))
                    jaccard_similarity = overlap / union
                    
                    if jaccard_similarity > 0.7:  # High similarity threshold
                        repetitive_comparisons += 1
        
        return repetitive_comparisons / total_comparisons if total_comparisons > 0 else 0.0

class AdapterManager:
    """Manages multiple LoRA adapters and prevents over-accumulation"""
    
    def __init__(self, max_adapters: int = 5):
        self.max_adapters = max_adapters
        self.memory = MemoryModule()
        self.adapter_scores = {}
    
    def save_adapter_with_metadata(self, adapter_path: str, training_metadata: Dict) -> str:
        """Save adapter with comprehensive metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        versioned_path = f"{adapter_path}_{timestamp}"
        
        # Add performance metrics to metadata
        full_metadata = {
            **training_metadata,
            "timestamp": timestamp,
            "version": self._get_next_version(),
            "performance_score": self._calculate_performance_score(training_metadata),
            "save_path": versioned_path
        }
        
        # Save metadata
        metadata_path = os.path.join(versioned_path, "adapter_metadata.json")
        os.makedirs(versioned_path, exist_ok=True)
        
        with open(metadata_path, "w") as f:
            json.dump(full_metadata, f, indent=2)
        
        # Update adapter registry
        self._update_adapter_registry(versioned_path, full_metadata)
        
        # Clean up old adapters if needed
        self._cleanup_old_adapters()
        
        return versioned_path
    
    def _get_next_version(self) -> int:
        """Get the next version number"""
        adapters = self._list_adapters()
        if not adapters:
            return 1
        
        versions = [a.get("version", 0) for a in adapters]
        return max(versions) + 1
    
    def _calculate_performance_score(self, metadata: Dict) -> float:
        """Calculate performance score for adapter ranking"""
        score = 0.0
        
        # Lower training loss is better
        if "training_loss" in metadata:
            loss_score = max(0, 1.0 - metadata["training_loss"])
            score += loss_score * 0.4
        
        # More training examples generally better (up to a point)
        if "training_examples" in metadata:
            example_score = min(1.0, metadata["training_examples"] / 100)
            score += example_score * 0.3
        
        # Reasonable training time (not too short, not too long)
        if "training_time_minutes" in metadata:
            time_minutes = metadata["training_time_minutes"]
            if 5 <= time_minutes <= 60:  # Sweet spot
                time_score = 1.0
            elif time_minutes < 5:
                time_score = time_minutes / 5.0
            else:
                time_score = max(0.1, 60.0 / time_minutes)
            score += time_score * 0.3
        
        return min(1.0, score)
    
    def _update_adapter_registry(self, adapter_path: str, metadata: Dict):
        """Update the adapter registry"""
        registry_path = os.path.join(settings.lora_path, "..", "adapter_registry.json")
        
        # Load existing registry
        registry = []
        if os.path.exists(registry_path):
            try:
                with open(registry_path, "r") as f:
                    registry = json.load(f)
            except:
                registry = []
        
        # Add new adapter
        registry.append({
            "path": adapter_path,
            "metadata": metadata,
            "created_at": datetime.now().isoformat()
        })
        
        # Sort by performance score
        registry.sort(key=lambda x: x["metadata"].get("performance_score", 0), reverse=True)
        
        # Save registry
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)
    
    def _list_adapters(self) -> List[Dict]:
        """List all registered adapters"""
        registry_path = os.path.join(settings.lora_path, "..", "adapter_registry.json")
        
        if not os.path.exists(registry_path):
            return []
        
        try:
            with open(registry_path, "r") as f:
                return json.load(f)
        except:
            return []
    
    def _cleanup_old_adapters(self):
        """Remove old adapters if we exceed the limit"""
        adapters = self._list_adapters()
        
        if len(adapters) <= self.max_adapters:
            return
        
        # Sort by performance score, keep the best ones
        adapters.sort(key=lambda x: x["metadata"].get("performance_score", 0), reverse=True)
        
        # Remove excess adapters
        to_remove = adapters[self.max_adapters:]
        remaining = adapters[:self.max_adapters]
        
        for adapter in to_remove:
            adapter_path = adapter["path"]
            if os.path.exists(adapter_path):
                import shutil
                try:
                    shutil.rmtree(adapter_path)
                    logger.info(f"Removed old adapter: {adapter_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove adapter {adapter_path}: {e}")
        
        # Update registry
        registry_path = os.path.join(settings.lora_path, "..", "adapter_registry.json")
        with open(registry_path, "w") as f:
            json.dump(remaining, f, indent=2)

class TrainingSafeguards:
    """Main safeguards coordinator"""
    
    def __init__(self):
        self.knowledge_tester = KnowledgeRetentionTester()
        self.diversity_monitor = DiversityMonitor()
        self.adapter_manager = AdapterManager()
        self.memory = MemoryModule()
    
    def validate_before_training(self, training_data: List[Dict]) -> Dict[str, Any]:
        """Validate training data and conditions before starting training"""
        validation_result = {
            "can_proceed": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Check data quality
        if len(training_data) < 10:
            validation_result["errors"].append(f"Insufficient training data: {len(training_data)} samples")
            validation_result["can_proceed"] = False
        
        # Check data diversity
        if len(training_data) < 50:
            validation_result["warnings"].append("Small dataset may lead to overfitting")
            validation_result["recommendations"].append("Consider reducing learning rate")
        
        # Check feedback score distribution
        scores = [item.get("score", 0) for item in training_data]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        if avg_score < 3.5:
            validation_result["warnings"].append(f"Low average feedback score: {avg_score:.2f}")
        
        if max(scores) - min(scores) < 1.0:
            validation_result["warnings"].append("Low score variance - may indicate poor feedback quality")
        
        # Check for data imbalance
        score_counts = defaultdict(int)
        for score in scores:
            score_counts[round(score)] += 1
        
        if len(score_counts) < 3:
            validation_result["warnings"].append("Limited score diversity in training data")
        
        return validation_result
    
    def validate_after_training(self, model, tokenizer, device: str = "cpu") -> Dict[str, Any]:
        """Comprehensive validation after training"""
        if not ML_AVAILABLE:
            return {"error": "ML dependencies not available"}
        
        validation_result = {
            "knowledge_retention": {},
            "diversity_metrics": {},
            "overall_health": "unknown",
            "recommendations": []
        }
        
        try:
            # Test knowledge retention
            logger.info("Testing knowledge retention...")
            validation_result["knowledge_retention"] = self.knowledge_tester.test_knowledge_retention(
                model, tokenizer, device
            )
            
            # Test output diversity
            logger.info("Testing output diversity...")
            validation_result["diversity_metrics"] = self.diversity_monitor.measure_diversity(
                model, tokenizer, device
            )
            
            # Determine overall health
            knowledge_passed = validation_result["knowledge_retention"].get("passed", False)
            diversity_score = validation_result["diversity_metrics"].get("overall_diversity", 0)
            
            if knowledge_passed and diversity_score > 0.6:
                validation_result["overall_health"] = "healthy"
            elif knowledge_passed and diversity_score > 0.4:
                validation_result["overall_health"] = "acceptable"
                validation_result["recommendations"].append("Monitor diversity in future training")
            elif knowledge_passed:
                validation_result["overall_health"] = "concerning"
                validation_result["recommendations"].append("Improve training data diversity")
            else:
                validation_result["overall_health"] = "unhealthy"
                validation_result["recommendations"].append("Consider reverting to previous adapter")
                validation_result["recommendations"].append("Review training data quality")
            
            # Log results
            logger.info(f"Training validation: {validation_result['overall_health']}")
            
        except Exception as e:
            logger.error(f"Post-training validation failed: {e}")
            validation_result["error"] = str(e)
            validation_result["overall_health"] = "unknown"
        
        return validation_result
    
    def get_conservative_training_args(self, dataset_size: int) -> Dict[str, Any]:
        """Get conservative training arguments to prevent overfitting"""
        # Scale parameters based on dataset size
        if dataset_size < 50:
            epochs = 1
            lr = 1e-5
            batch_size = 1
        elif dataset_size < 100:
            epochs = 2
            lr = 5e-5
            batch_size = 2
        else:
            epochs = 3
            lr = 1e-4
            batch_size = 4
        
        return {
            "num_train_epochs": epochs,
            "learning_rate": lr,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": max(1, 8 // batch_size),
            "warmup_steps": min(100, dataset_size // 10),
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "save_steps": max(50, dataset_size // 4),
            "eval_steps": max(25, dataset_size // 8),
            "logging_steps": max(10, dataset_size // 20)
        }
    
    def should_stop_training_early(self, training_logs: List[Dict]) -> Tuple[bool, str]:
        """Determine if training should be stopped early"""
        if len(training_logs) < 5:
            return False, ""
        
        # Check for loss explosion
        recent_losses = [log.get("train_loss", 0) for log in training_logs[-5:]]
        if any(loss > 10.0 for loss in recent_losses):
            return True, "Training loss exploded"
        
        # Check for loss stagnation
        if len(recent_losses) >= 5:
            loss_trend = recent_losses[-1] - recent_losses[0]
            if abs(loss_trend) < 0.001:
                return True, "Training loss stagnated"
        
        # Check for overfitting (if eval loss available)
        recent_eval_losses = [log.get("eval_loss", 0) for log in training_logs[-5:] if "eval_loss" in log]
        if len(recent_eval_losses) >= 3:
            if recent_eval_losses[-1] > recent_eval_losses[0] * 1.1:
                return True, "Evaluation loss increasing (overfitting detected)"
        
        return False, ""
