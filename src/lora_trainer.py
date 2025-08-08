import os
import json
import torch
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import guards for ML dependencies
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
        Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
    )
    from peft import (
        LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    )
    from datasets import Dataset
    import torch.nn.functional as F
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML dependencies not available for LoRA training")

try:
    from .config import settings  # type: ignore
except Exception:  # pragma: no cover
    from config import settings  # type: ignore
try:
    from .memory_module import MemoryModule  # type: ignore
except Exception:  # pragma: no cover
    from memory_module import MemoryModule  # type: ignore

# Handle import for training safeguards
try:
    from .training_safeguards import TrainingSafeguards
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.training_safeguards import TrainingSafeguards

class LoRATrainer:
    def __init__(self):
        self.memory = MemoryModule()
        self.safeguards = TrainingSafeguards()
        self.model = None
        self.tokenizer = None
        self.training_data = []
        self.device = "cuda" if ML_AVAILABLE and torch and torch.cuda.is_available() and not settings.force_cpu else "cpu"
        
        # Only initialize LoRA config if ML libraries are available
        if ML_AVAILABLE:
            try:
                # Conservative LoRA configuration to prevent overfitting
                self.lora_config = LoraConfig(
                    r=8,  # Reduced rank for more conservative training
                    lora_alpha=16,  # Reduced alpha
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
                    lora_dropout=0.2,  # Increased dropout for regularization
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize LoRA config: {e}")
                self.lora_config = None
        else:
            self.lora_config = None
            logger.warning("ML dependencies not available, LoRA training will be disabled")
        
        # Conservative training configuration to prevent overfitting
        base_args = {
            "output_dir": os.path.join(settings.lora_path, "training_output"),
            "per_device_eval_batch_size": 1,
            "fp16": True if self.device == "cuda" else False,
            "evaluation_strategy": "steps",
            "save_total_limit": 2,  # Keep fewer checkpoints
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "report_to": None,
            "remove_unused_columns": False,
            "dataloader_pin_memory": False,
        }
        
        self.training_args = TrainingArguments(**base_args)

    def prepare_training_data(self, min_score: float = 3.0, max_samples: int = 1000) -> List[Dict]:
        """Prepare training data from memory with feedback filtering"""
        logger.info("Preparing training data from memory...")
        
        # Get all sessions with feedback
        feedback_entries = self.memory.get_feedback_entries(min_score=min_score)
        
        training_examples = []
        processed_sessions = set()
        
        for feedback in feedback_entries:
            if feedback.session_id in processed_sessions:
                continue
                
            # Get all entries for this session
            session_entries = self.memory.read_by_session(feedback.session_id)
            
            # Find prompt and output pairs
            prompt_entry = None
            output_entry = None
            
            for entry in session_entries:
                if entry.entry_type in ["prompt", "self_prompt", "focused_prompt"]:
                    prompt_entry = entry
                elif entry.entry_type in ["output", "self_output", "focused_output"]:
                    output_entry = entry
                    
            if prompt_entry and output_entry:
                # Create training example in chat format
                training_example = {
                    "input_text": prompt_entry.content,
                    "output_text": output_entry.content,
                    "score": feedback.score,
                    "session_id": feedback.session_id,
                    "timestamp": feedback.timestamp.isoformat()
                }
                
                training_examples.append(training_example)
                processed_sessions.add(feedback.session_id)
                
                if len(training_examples) >= max_samples:
                    break
        
        logger.info(f"Prepared {len(training_examples)} training examples from {len(processed_sessions)} sessions")
        return training_examples

    def format_training_data(self, examples: List[Dict]) -> List[str]:
        """Format training data into chat format for TinyLlama"""
        formatted_texts = []
        
        for example in examples:
            # Use TinyLlama's chat format
            chat_text = f"<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n{example['input_text']}</s>\n<|assistant|>\n{example['output_text']}</s>"
            formatted_texts.append(chat_text)
            
        return formatted_texts

    def create_dataset(self, texts: List[str], max_length: int = 512) -> Dataset:
        """Create tokenized dataset for training"""
        if not ML_AVAILABLE:
            raise RuntimeError("ML dependencies not available")
            
        # Initialize tokenizer if not already done
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Tokenize texts
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt"
            )
        
        # Create dataset
        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset

    def load_base_model(self):
        """Load and prepare the base model for LoRA training"""
        if not ML_AVAILABLE:
            raise RuntimeError("ML dependencies not available")
            
        logger.info(f"Loading base model: {settings.model_name}")
        
        # Load model with specific settings for LoRA training
        model = AutoModelForCausalLM.from_pretrained(
            settings.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else "cpu",
            trust_remote_code=True,
            use_cache=False  # Disable cache for training
        )
        
        # Prepare model for k-bit training if using GPU
        if self.device == "cuda":
            model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA
        logger.info("Applying LoRA configuration...")
        model = get_peft_model(model, self.lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        self.model = model
        return model

    def train_lora(self, min_feedback_score: float = 3.0, max_samples: int = 500, force_training: bool = False) -> Dict:
        """Train LoRA adapter on feedback data with comprehensive safeguards"""
        if not ML_AVAILABLE:
            return {"error": "ML dependencies not available", "status": "failed"}
            
        try:
            logger.info("Starting LoRA training session with safeguards...")
            start_time = datetime.now()
            
            # Prepare training data
            training_examples = self.prepare_training_data(min_feedback_score, max_samples)
            
            if len(training_examples) < 10 and not force_training:
                return {
                    "error": f"Insufficient training data: {len(training_examples)} examples (minimum 10 required)",
                    "status": "failed",
                    "recommendation": "Collect more feedback or use force_training=True"
                }
            
            # Pre-training validation
            logger.info("Running pre-training validation...")
            validation = self.safeguards.validate_before_training(training_examples)
            
            if not validation["can_proceed"] and not force_training:
                return {
                    "error": "Pre-training validation failed",
                    "status": "failed",
                    "validation_errors": validation["errors"],
                    "validation_warnings": validation["warnings"]
                }
            
            # Log validation warnings
            for warning in validation["warnings"]:
                logger.warning(f"Training warning: {warning}")
            
            # Get conservative training arguments based on dataset size
            conservative_args = self.safeguards.get_conservative_training_args(len(training_examples))
            
            # Update training arguments with conservative settings
            for key, value in conservative_args.items():
                setattr(self.training_args, key, value)
            
            logger.info(f"Using conservative training settings: epochs={conservative_args['num_train_epochs']}, "
                       f"lr={conservative_args['learning_rate']}, batch_size={conservative_args['per_device_train_batch_size']}")
            
            # Format data for training
            formatted_texts = self.format_training_data(training_examples)
            
            # Create dataset
            logger.info("Creating tokenized dataset...")
            dataset = self.create_dataset(formatted_texts)
            
            # Split dataset (80% train, 20% eval) with minimum sizes
            test_size = max(0.2, min(0.3, 5.0 / len(dataset)))  # At least 5 samples for eval
            split_dataset = dataset.train_test_split(test_size=test_size)
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
            
            logger.info(f"Training samples: {len(train_dataset)}, Evaluation samples: {len(eval_dataset)}")
            
            # Load and prepare model
            model = self.load_base_model()
            
            # Store original model for comparison
            if hasattr(model, 'base_model'):
                original_model = model.base_model
            else:
                original_model = model
            
            # Initialize tokenizer
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Test knowledge retention before training (baseline)
            logger.info("Testing baseline knowledge retention...")
            baseline_knowledge = self.safeguards.knowledge_tester.test_knowledge_retention(
                original_model, self.tokenizer, self.device
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8 if self.device == "cuda" else None
            )
            
            # Custom trainer class with early stopping safeguards
            class SafeguardedTrainer(Trainer):
                def __init__(self, safeguards_instance, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.safeguards = safeguards_instance
                    self.training_logs = []
                
                def log(self, logs):
                    super().log(logs)
                    self.training_logs.append(logs)
                    
                    # Check if we should stop early
                    should_stop, reason = self.safeguards.should_stop_training_early(self.training_logs)
                    if should_stop:
                        logger.warning(f"Early stopping triggered: {reason}")
                        self.should_training_stop = True
            
            # Create trainer with safeguards
            trainer = SafeguardedTrainer(
                safeguards_instance=self.safeguards,
                model=model,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)]
            )
            
            # Start training
            logger.info("Starting training with safeguards...")
            training_result = trainer.train()
            
            # Post-training validation
            logger.info("Running post-training validation...")
            post_validation = self.safeguards.validate_after_training(model, self.tokenizer, self.device)
            
            # Check if training was successful
            training_healthy = post_validation.get("overall_health") in ["healthy", "acceptable"]
            
            if not training_healthy and not force_training:
                logger.warning("Post-training validation indicates unhealthy model state")
                # Optionally revert to previous adapter here
            
            # Save the trained adapter with comprehensive metadata
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            adapter_path = os.path.join(settings.lora_path, f"adapter_{timestamp}")
            os.makedirs(adapter_path, exist_ok=True)
            
            logger.info(f"Saving LoRA adapter to {adapter_path}")
            model.save_pretrained(adapter_path)
            self.tokenizer.save_pretrained(adapter_path)
            
            # Enhanced training metadata
            metadata = {
                "training_examples": len(training_examples),
                "min_feedback_score": min_feedback_score,
                "training_loss": float(training_result.training_loss),
                "training_steps": training_result.global_step,
                "model_name": settings.model_name,
                "lora_config": {
                    "r": self.lora_config.r,
                    "lora_alpha": self.lora_config.lora_alpha,
                    "lora_dropout": self.lora_config.lora_dropout,
                },
                "training_time_minutes": (datetime.now() - start_time).total_seconds() / 60,
                "conservative_settings": conservative_args,
                "pre_training_validation": validation,
                "post_training_validation": post_validation,
                "baseline_knowledge": baseline_knowledge,
                "training_healthy": training_healthy,
                "safeguards_version": "1.0"
            }
            
            # Save metadata and register adapter
            with open(os.path.join(adapter_path, "training_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Use adapter manager for versioning and cleanup
            versioned_path = self.safeguards.adapter_manager.save_adapter_with_metadata(
                adapter_path, metadata
            )
            
            # Update the default adapter path only if training was healthy
            if training_healthy or force_training:
                latest_link = settings.lora_path
                if os.path.islink(latest_link):
                    os.unlink(latest_link)
                elif os.path.exists(latest_link):
                    import shutil
                    shutil.rmtree(latest_link)
                    
                os.symlink(versioned_path, latest_link)
                logger.info("Updated default adapter to new version")
            else:
                logger.warning("Keeping previous adapter due to validation concerns")
            
            logger.info(f"LoRA training completed in {metadata['training_time_minutes']:.2f} minutes")
            
            # Store comprehensive training result in memory
            result_summary = (
                f"LORA_TRAINING_COMPLETED: {len(training_examples)} examples, "
                f"loss={training_result.training_loss:.4f}, "
                f"health={post_validation.get('overall_health', 'unknown')}, "
                f"knowledge_retained={post_validation.get('knowledge_retention', {}).get('passed', False)}"
            )
            
            self.memory.write(
                content=result_summary,
                entry_type="training_result",
                session_id=f"lora_training_{timestamp}",
                score=5.0 - training_result.training_loss if training_healthy else 2.0
            )
            
            return {
                "status": "completed",
                "adapter_path": versioned_path,
                "training_examples": len(training_examples),
                "training_loss": float(training_result.training_loss),
                "training_time_minutes": metadata['training_time_minutes'],
                "training_healthy": training_healthy,
                "validation_results": post_validation,
                "recommendations": post_validation.get("recommendations", []),
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"LoRA training failed: {e}", exc_info=True)
            
            # Store failure in memory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.memory.write(
                content=f"LORA_TRAINING_FAILED: {str(e)}",
                entry_type="training_result",
                session_id=f"lora_training_failed_{timestamp}",
                score=0.0
            )
            
            return {
                "status": "failed",
                "error": str(e),
                "training_examples": len(getattr(self, 'training_examples', [])),
                "timestamp": timestamp
            }

    def evaluate_adapter(self, adapter_path: str, test_prompts: List[str] = None) -> Dict:
        """Evaluate a trained LoRA adapter"""
        if not ML_AVAILABLE:
            return {"error": "ML dependencies not available"}
            
        try:
            # Default test prompts
            if test_prompts is None:
                test_prompts = [
                    "What is artificial intelligence?",
                    "Explain machine learning in simple terms.",
                    "How do neural networks work?",
                    "What are the benefits of renewable energy?",
                    "Describe the process of photosynthesis."
                ]
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                settings.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else "cpu",
            )
            
            # Load LoRA adapter
            from peft import PeftModel
            model = PeftModel.from_pretrained(base_model, adapter_path)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(adapter_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Generate responses
            results = []
            for prompt in test_prompts:
                formatted_prompt = f"<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
                
                inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.replace(formatted_prompt, "").strip()
                
                results.append({
                    "prompt": prompt,
                    "response": response
                })
            
            return {
                "status": "completed",
                "adapter_path": adapter_path,
                "test_results": results
            }
            
        except Exception as e:
            logger.error(f"Adapter evaluation failed: {e}")
            return {"status": "failed", "error": str(e)}

    def get_training_status(self) -> Dict:
        """Get current training status and statistics with health metrics"""
        try:
            # Get training data statistics
            all_feedback = self.memory.get_feedback_entries()
            good_feedback = self.memory.get_feedback_entries(min_score=3.0)
            training_data = self.memory.get_training_data()
            
            # Get recent training results
            training_results = self.memory.read_by_type("training_result", limit=10)
            
            # Check if adapter exists
            adapter_exists = os.path.exists(settings.lora_path) and os.path.isdir(settings.lora_path)
            
            # Get adapter health information
            adapter_health = self._check_adapter_health()
            
            # Calculate data quality metrics
            data_quality = self._calculate_data_quality(all_feedback)
            
            return {
                "total_feedback_entries": len(all_feedback),
                "good_feedback_entries": len(good_feedback),
                "available_training_pairs": len(training_data),
                "ready_for_training": len(good_feedback) >= 10,
                "adapter_exists": adapter_exists,
                "adapter_path": settings.lora_path,
                "adapter_health": adapter_health,
                "data_quality": data_quality,
                "recent_training_sessions": len(training_results),
                "ml_available": ML_AVAILABLE,
                "safeguards_enabled": True
            }
            
        except Exception as e:
            logger.error(f"Failed to get training status: {e}")
            return {"error": str(e)}
    
    def _check_adapter_health(self) -> Dict[str, Any]:
        """Check the health of the current adapter"""
        health_info = {
            "status": "unknown",
            "last_validation": None,
            "knowledge_retention": None,
            "diversity_score": None
        }
        
        try:
            if not os.path.exists(settings.lora_path):
                health_info["status"] = "no_adapter"
                return health_info
            
            # Check for metadata file
            metadata_path = os.path.join(settings.lora_path, "training_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                
                post_validation = metadata.get("post_training_validation", {})
                health_info["status"] = post_validation.get("overall_health", "unknown")
                health_info["knowledge_retention"] = post_validation.get("knowledge_retention", {}).get("passed", None)
                health_info["diversity_score"] = post_validation.get("diversity_metrics", {}).get("overall_diversity", None)
                health_info["last_validation"] = metadata.get("training_time_minutes", None)
            else:
                health_info["status"] = "legacy_adapter"
                
        except Exception as e:
            logger.warning(f"Failed to check adapter health: {e}")
            health_info["status"] = "error"
            health_info["error"] = str(e)
        
        return health_info
    
    def _calculate_data_quality(self, feedback_entries: List) -> Dict[str, Any]:
        """Calculate data quality metrics"""
        if not feedback_entries:
            return {"score": 0.0, "issues": ["No feedback data available"]}
        
        scores = [f.score for f in feedback_entries if f.score is not None]
        
        if not scores:
            return {"score": 0.0, "issues": ["No scored feedback"]}
        
        quality_metrics = {
            "total_entries": len(feedback_entries),
            "scored_entries": len(scores),
            "average_score": sum(scores) / len(scores),
            "score_variance": np.var(scores) if len(scores) > 1 else 0,
            "high_quality_ratio": len([s for s in scores if s >= 4.0]) / len(scores),
            "issues": []
        }
        
        # Identify data quality issues
        if quality_metrics["average_score"] < 3.0:
            quality_metrics["issues"].append("Low average feedback score")
        
        if quality_metrics["score_variance"] < 0.5:
            quality_metrics["issues"].append("Low score variance - poor feedback diversity")
        
        if quality_metrics["high_quality_ratio"] < 0.2:
            quality_metrics["issues"].append("Too few high-quality examples")
        
        if len(scores) < 50:
            quality_metrics["issues"].append("Insufficient feedback volume")
        
        # Calculate overall quality score
        quality_score = 0.0
        quality_score += min(1.0, quality_metrics["average_score"] / 4.0) * 0.3  # Average score component
        quality_score += min(1.0, quality_metrics["score_variance"]) * 0.2  # Variance component
        quality_score += quality_metrics["high_quality_ratio"] * 0.3  # High quality ratio
        quality_score += min(1.0, len(scores) / 100) * 0.2  # Volume component
        
        quality_metrics["score"] = quality_score
        
        return quality_metrics
    
    def validate_current_adapter(self) -> Dict[str, Any]:
        """Validate the currently active adapter"""
        if not ML_AVAILABLE:
            return {"error": "ML dependencies not available"}
        
        try:
            if not os.path.exists(settings.lora_path):
                return {"error": "No adapter found"}
            
            # Load model and run validation
            logger.info("Loading current adapter for validation...")
            base_model = AutoModelForCausalLM.from_pretrained(
                settings.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else "cpu",
            )
            
            from peft import PeftModel
            model = PeftModel.from_pretrained(base_model, settings.lora_path)
            
            tokenizer = AutoTokenizer.from_pretrained(settings.lora_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Run comprehensive validation
            validation_results = self.safeguards.validate_after_training(model, tokenizer, self.device)
            
            logger.info(f"Adapter validation completed: {validation_results.get('overall_health', 'unknown')}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Adapter validation failed: {e}")
            return {"error": str(e)}
