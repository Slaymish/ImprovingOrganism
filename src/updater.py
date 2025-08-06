import os
import json
from typing import List, Tuple
from datetime import datetime, timedelta
import logging

try:
    import torch
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from torch.utils.data import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from .llm_wrapper import LLMWrapper
from .memory_module import MemoryModule
from .critic_module import CriticModule
from .config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if ML_AVAILABLE:
    class FeedbackDataset(Dataset):
        """Dataset for fine-tuning on feedback data"""
        
        def __init__(self, training_pairs: List[Tuple[str, str, float]], tokenizer, max_length=512):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.data = []
            
            for prompt, output, score in training_pairs:
                # Format as instruction-following pairs
                text = f"<|user|>\n{prompt}\n<|assistant|>\n{output}<|endoftext|>"
                encoding = tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'
                )
                self.data.append({
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'score': score
                })
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
else:
    # Mock dataset for development
    class FeedbackDataset:
        def __init__(self, training_pairs, tokenizer=None, max_length=512):
            self.data = training_pairs
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]

class Updater:
    def __init__(self):
        self.memory = MemoryModule()
        self.critic = CriticModule()
        self.last_training_time = None
        
    def should_retrain(self) -> bool:
        """Determine if retraining is needed based on feedback volume and time"""
        feedback_entries = self.memory.get_feedback_entries()
        
        # Check if we have enough feedback
        if len(feedback_entries) < settings.feedback_threshold:
            logger.info(f"Not enough feedback entries: {len(feedback_entries)} < {settings.feedback_threshold}")
            return False
            
        # Check if enough time has passed since last training
        if self.last_training_time:
            time_since_training = datetime.utcnow() - self.last_training_time
            if time_since_training < timedelta(hours=6):  # Wait at least 6 hours
                logger.info(f"Too soon since last training: {time_since_training}")
                return False
        
        # Check if we have enough good feedback (score >= 3.0)
        good_feedback = [f for f in feedback_entries if f.score and f.score >= 3.0]
        if len(good_feedback) < settings.feedback_threshold * 0.3:  # At least 30% should be good
            logger.info(f"Not enough good feedback: {len(good_feedback)}")
            return False
            
        return True
    
    def prepare_training_data(self) -> List[Tuple[str, str, float]]:
        """Extract and prepare training data from memory"""
        training_pairs = self.memory.get_training_data(limit=200)
        
        if not training_pairs:
            logger.warning("No training pairs found in memory")
            return []
            
        # Filter and clean training pairs
        cleaned_pairs = []
        for prompt, output, score in training_pairs:
            if len(prompt.strip()) > 10 and len(output.strip()) > 10 and score >= 3.0:
                cleaned_pairs.append((prompt.strip(), output.strip(), score))
        
        logger.info(f"Prepared {len(cleaned_pairs)} training pairs")
        return cleaned_pairs
    
    def fine_tune_lora(self, training_pairs: List[Tuple[str, str, float]]) -> bool:
        """Fine-tune the LoRA adapter on feedback data"""
        if not ML_AVAILABLE:
            logger.info("⚠️  ML dependencies not available. Skipping training (mock mode).")
            # Mock training for development
            self.memory.write(
                f"MOCK_TRAINING: {len(training_pairs)} samples (development mode)",
                entry_type="training",
                session_id="updater"
            )
            self.last_training_time = datetime.utcnow()
            return True
            
        try:
            logger.info("Starting LoRA fine-tuning...")
            
            # Initialize LLM wrapper to get tokenizer and model
            llm = LLMWrapper()
            tokenizer = llm.tokenizer
            model = llm.model
            
            if tokenizer is None or model is None:
                logger.warning("Model not available, using mock training")
                self.memory.write(
                    f"MOCK_TRAINING: {len(training_pairs)} samples (model unavailable)",
                    entry_type="training",
                    session_id="updater"
                )
                self.last_training_time = datetime.utcnow()
                return True
            
            # Prepare model for training
            model = prepare_model_for_kbit_training(model)
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            model = get_peft_model(model, lora_config)
            
            # Create dataset
            dataset = FeedbackDataset(training_pairs, tokenizer)
            
            if len(dataset) == 0:
                logger.warning("Empty dataset, skipping training")
                return False
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"./training_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                num_train_epochs=2,
                per_device_train_batch_size=settings.batch_size,
                gradient_accumulation_steps=4,
                warmup_steps=10,
                logging_steps=10,
                save_steps=100,
                evaluation_strategy="no",
                save_strategy="steps",
                load_best_model_at_end=False,
                report_to="none",
                remove_unused_columns=False,
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
            
            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
            )
            
            # Train
            logger.info("Starting training...")
            trainer.train()
            
            # Save the new adapter
            new_adapter_path = f"{settings.lora_path}_updated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model.save_pretrained(new_adapter_path)
            
            # Update config to use new adapter
            # In a real implementation, you'd want to atomically update this
            logger.info(f"Saved new adapter to: {new_adapter_path}")
            
            # Log training completion
            self.memory.write(
                f"TRAINING_COMPLETED: {len(training_pairs)} samples, saved to {new_adapter_path}",
                entry_type="training",
                session_id="updater"
            )
            
            self.last_training_time = datetime.utcnow()
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.memory.write(
                f"TRAINING_FAILED: {str(e)}",
                entry_type="error",
                session_id="updater"
            )
            return False
    
    def evaluate_current_model(self) -> float:
        """Evaluate current model performance on recent feedback"""
        try:
            recent_feedback = self.memory.get_feedback_entries()[-20:]  # Last 20 feedback entries
            
            if not recent_feedback:
                return 3.0  # Neutral score
                
            scores = [f.score for f in recent_feedback if f.score is not None]
            
            if not scores:
                return 3.0
                
            avg_score = sum(scores) / len(scores)
            logger.info(f"Current model average score: {avg_score:.2f}")
            
            return avg_score
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 3.0
    
    def run_update_cycle(self):
        """Main update cycle: check feedback, retrain if needed"""
        logger.info("Starting update cycle...")
        
        # Check if retraining is needed
        if not self.should_retrain():
            logger.info("Retraining not needed at this time")
            return
        
        # Evaluate current performance
        current_performance = self.evaluate_current_model()
        
        # Prepare training data
        training_pairs = self.prepare_training_data()
        
        if not training_pairs:
            logger.info("No training data available")
            return
        
        # Perform fine-tuning
        success = self.fine_tune_lora(training_pairs)
        
        if success:
            logger.info("LoRA fine-tuning completed successfully")
            
            # Log the update
            self.memory.write(
                f"MODEL_UPDATED: Previous avg score: {current_performance:.2f}, Training samples: {len(training_pairs)}",
                entry_type="update",
                session_id="updater",
                score=current_performance
            )
        else:
            logger.error("LoRA fine-tuning failed")

def main():
    """Main updater function"""
    updater = Updater()
    
    logger.info("Updater starting...")
    
    # Run one update cycle
    updater.run_update_cycle()
    
    logger.info("Updater cycle completed")

if __name__ == "__main__":
    main()
