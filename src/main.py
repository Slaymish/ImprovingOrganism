from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Union
import uuid
import logging
from datetime import datetime
from .llm_wrapper import LLMWrapper
from .memory_module import MemoryModule
from .critic_module import CriticModule
from .self_learning import SelfLearningModule
from .lora_trainer import LoRATrainer
from .metrics import metrics, time_block

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ImprovingOrganism API", version="1.0.0")

# Initialize components
try:
    memory = MemoryModule()
    critic = CriticModule()
    logger.info("Core components (memory, critic) initialized")
    
    # Initialize LLM in a separate step (this takes time)
    try:
        logger.info("Loading LLM wrapper...")
        llm = LLMWrapper()
        logger.info("LLM wrapper initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LLM wrapper: {e}")
        llm = None
    
    # Initialize optional components with individual error handling
    try:
        self_learner = SelfLearningModule()
        logger.info("Self-learner initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize self-learner: {e}")
        self_learner = None
    
    try:
        lora_trainer = LoRATrainer()
        logger.info("LoRA trainer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LoRA trainer: {e}")
        lora_trainer = None
        
    logger.info("All components initialization completed")
except Exception as e:
    logger.error(f"Failed to initialize core components: {e}", exc_info=True)
    # Create mock components for development
    llm = None
    memory = MemoryModule()
    critic = CriticModule()
    self_learner = None
    lora_trainer = None

class Prompt(BaseModel):
    text: str
    session_id: Optional[str] = None

class Feedback(BaseModel):
    prompt: str
    output: str
    score: float
    comment: Optional[str] = None
    session_id: Optional[str] = None

class GenerationResponse(BaseModel):
    output: str
    session_id: str
    score: Optional[float] = None

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class SelfLearnRequest(BaseModel):
    iterations: int
    topic: Optional[str] = None

class TrainRequest(BaseModel):
    mode: str = "lora"  # Always LoRA training
    session_id: Optional[str] = None
    min_feedback_score: Optional[float] = 3.0
    max_samples: Optional[int] = 500
    force_retrain: Optional[bool] = False

class FeedbackRequest(BaseModel):
    response_id: str
    score: float  # Will accept int or float, as pydantic auto-converts
    feedback: Optional[str] = None
    session_id: Optional[str] = None

@app.get("/")
def root():
    return {"message": "ImprovingOrganism API", "status": "running"}

@app.get("/health")
def health_check():
    """Comprehensive health check of all components"""
    health_status = {
        "api": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "llm": llm is not None,
            "memory": memory is not None,
            "critic": critic is not None,
            "self_learner": self_learner is not None,
            "lora_trainer": lora_trainer is not None
        },
        "details": {
            "llm_status": "loaded" if llm else "not_available",
            "memory_entries": len(memory.read_all()) if memory else 0,
            "ml_available": getattr(lora_trainer, 'lora_config', None) is not None if lora_trainer else False
        }
    }
    
    # Overall system health
    critical_components = ["memory", "critic"]
    optional_components = ["llm", "self_learner", "lora_trainer"]
    
    critical_healthy = all(health_status["components"][comp] for comp in critical_components)
    optional_healthy = sum(health_status["components"][comp] for comp in optional_components)
    
    if critical_healthy:
        if optional_healthy >= 2:
            health_status["overall"] = "healthy"
        elif optional_healthy >= 1:
            health_status["overall"] = "degraded"
        else:
            health_status["overall"] = "limited"
    else:
        health_status["overall"] = "unhealthy"
    
    return health_status

@app.post("/query/simple")
def simple_query(request: QueryRequest):
    """Simple query endpoint that returns a fast mock response for testing"""
    session_id = request.session_id or str(uuid.uuid4())
    
    # Store the prompt
    memory.write(
        content=request.query,
        entry_type="prompt",
        session_id=session_id
    )
    
    # Return a quick mock response
    response = f"Quick test response to: {request.query}"
    
    # Store the output
    memory.write(
        content=response,
        entry_type="output",
        session_id=session_id
    )
    
    return {
        "response": response,
        "session_id": session_id,
        "metadata": {
            "mode": "simple_test",
            "timestamp": datetime.now().isoformat()
        }
    }

def _build_retrieval_context(user_text: str, max_items: int = 5) -> dict:
    """Retrieve semantic (if available) context. Records metrics.
    Returns dict with keys: items (list[str]), semantic (bool), latency_s (float).
    """
    timer = time_block()
    items = []
    semantic_used = False
    try:
        results = memory.search_semantic(user_text, limit=max_items)
        if results:
            semantic_used = True
            items = [r.get('content', '') for r in results if r.get('content')]
    except Exception:
        items = []
    latency = timer()
    if metrics:
        metrics.record_retrieval(latency, len(items), semantic_used)
    return {"items": items, "semantic": semantic_used, "latency_s": latency}

@app.post("/query")
def query(request: QueryRequest):
    """Handle LLM queries from the dashboard"""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        retrieval = _build_retrieval_context(request.query, max_items=3)
        if retrieval["items"]:
            context_str = "\n".join([f"Context snippet: {c}" for c in retrieval["items"]])
            augmented_prompt = (
                "You are an adaptive assistant. Use retrieved context when relevant; "
                "ignore irrelevant lines.\n\nRetrieved Context:\n" + context_str +
                f"\n\nUser Query: {request.query}\nAnswer:"
            )
        else:
            augmented_prompt = request.query

        # Store the prompt
        memory.write(
            content=request.query,
            entry_type="prompt",
            session_id=session_id
        )
        
        # Generate output
        if llm:
            output = llm.generate(augmented_prompt)
            logger.info(f"LLM model used: {llm.model_name} at {llm.model_path}")
        else:
            # Mock output for development
            output = f"Mock response to: {request.query[:50]}..."
            logger.warning("Using mock LLM output")
        
        # Store the output
        memory.write(
            content=output,
            entry_type="output",
            session_id=session_id
        )
        
        # Get recent memory for scoring context
        recent_memory = memory
        
        # Score the output automatically
        auto_score = critic.score(request.query, output, recent_memory)
        
        # Store automatic scoring as internal feedback
        memory.write(
            content=f"AUTO_SCORE: {request.query} -> {output} | score={auto_score:.2f}",
            entry_type="internal_feedback",
            session_id=session_id,
            score=auto_score
        )
        
    logger.info(f"Generated response for session {session_id} with auto-score {auto_score:.2f} retrieval_items={len(retrieval['items'])} retrieval_semantic={retrieval['semantic']}")
        
        return {
            "response": output,
            "session_id": session_id,
            "metadata": {
                "auto_score": auto_score,
                "model_name": getattr(llm, 'model_name', 'mock') if llm else 'mock',
                "model_path": getattr(llm, 'model_path', 'none') if llm else 'none',
                "timestamp": memory.read_by_session(session_id)[-1].timestamp if memory.read_by_session(session_id) else None,
                "semantic_context_used": retrieval["semantic"],
                "retrieval_items": len(retrieval["items"]),
                "retrieval_latency_ms": round(retrieval["latency_s"] * 1000, 2)
            }
        }
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/self_learn")
def self_learn(request: SelfLearnRequest):
    """Start a self-learning session"""
    try:
        if not self_learner:
            raise HTTPException(status_code=503, detail="Self-learning module not available")
        
        if request.iterations < 1 or request.iterations > 20:
            raise HTTPException(status_code=400, detail="Iterations must be between 1 and 20")
        
        session_id = str(uuid.uuid4())
        logger.info(f"Starting self-learning session {session_id} with {request.iterations} iterations")
        
        # Start the self-learning session in background
        import threading
        
        def run_learning_session():
            try:
                if request.topic:
                    # Use topic-focused learning if provided
                    results = self_learner.conduct_focused_learning_session(request.iterations, request.topic)
                else:
                    # Use general self-learning
                    results = self_learner.conduct_self_learning_session(request.iterations)
                
                # Store completion status in memory
                memory.write(
                    content=f"SELF_LEARNING_COMPLETED: {session_id} - avg_score: {results.get('average_score', 0):.3f}",
                    entry_type="self_learning_status",
                    session_id=session_id,
                    score=results.get('average_score', 0)
                )
                logger.info(f"Self-learning session {session_id} completed successfully")
                
            except Exception as e:
                logger.error(f"Self-learning session {session_id} failed: {e}")
                memory.write(
                    content=f"SELF_LEARNING_FAILED: {session_id} - error: {str(e)}",
                    entry_type="self_learning_status",
                    session_id=session_id,
                    score=0.0
                )
        
        # Start the learning session in a separate thread
        learning_thread = threading.Thread(target=run_learning_session)
        learning_thread.daemon = True
        learning_thread.start()
        
        # Return immediately with session info
        return {
            "status": "started",
            "session_id": session_id,
            "iterations_requested": request.iterations,
            "progress": f"0/{request.iterations} iterations completed",
            "topic": request.topic,
            "message": f"Self-learning session started with {request.iterations} iterations",
            "estimated_duration_minutes": request.iterations * 0.5  # Rough estimate
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Self-learning session failed: {e}")
        raise HTTPException(status_code=500, detail=f"Self-learning failed: {str(e)}")

@app.get("/self_learn/status/{session_id}")
def get_self_learning_status(session_id: str):
    """Check the status of a self-learning session"""
    try:
        # Check for status entries in memory
        status_entries = memory.read_by_session(session_id)
        status_entry = None
        
        for entry in status_entries:
            if entry.entry_type == "self_learning_status":
                status_entry = entry
                break
        
        if status_entry:
            if "COMPLETED" in status_entry.content:
                return {
                    "status": "completed",
                    "session_id": session_id,
                    "message": "Self-learning session completed successfully",
                    "average_score": status_entry.score,
                    "details": status_entry.content
                }
            elif "FAILED" in status_entry.content:
                return {
                    "status": "failed",
                    "session_id": session_id,
                    "message": "Self-learning session failed",
                    "error": status_entry.content
                }
        
        # Check if there are any learning entries for this session
        learning_entries = [e for e in status_entries if e.entry_type in ["self_output", "focused_output"]]
        
        if learning_entries:
            return {
                "status": "in_progress",
                "session_id": session_id,
                "iterations_completed": len(learning_entries),
                "message": f"Self-learning in progress ({len(learning_entries)} iterations completed)"
            }
        else:
            return {
                "status": "not_found",
                "session_id": session_id,
                "message": "No self-learning session found with this ID"
            }
            
    except Exception as e:
        logger.error(f"Failed to get self-learning status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@app.post("/train")
def train(request: TrainRequest):
    """Start LoRA training using feedback data"""
    try:
        if not lora_trainer:
            raise HTTPException(status_code=503, detail="LoRA trainer not available")
        
        session_id = request.session_id or str(uuid.uuid4())
        
        logger.info(f"Starting LoRA training session {session_id}")
        
        # Check if we have enough training data (unless forced)
        training_status = lora_trainer.get_training_status()
        
        # Log data quality warnings
        data_quality = training_status.get("data_quality", {})
        if data_quality.get("issues"):
            logger.warning(f"Data quality issues detected: {data_quality['issues']}")
        
        if not request.force_retrain and not training_status.get("ready_for_training", False):
            return {
                "status": "insufficient_data",
                "session_id": session_id,
                "message": f"Insufficient training data. Need at least 10 good feedback entries, have {training_status.get('good_feedback_entries', 0)}",
                "training_data_available": training_status.get("available_training_pairs", 0),
                "good_feedback_entries": training_status.get("good_feedback_entries", 0),
                "data_quality": data_quality,
                "recommendation": "Collect more feedback with scores >= 3.0 before training",
                "suggestion": "Set force_retrain=true to override this check"
            }
        
        # Determine if we should run training synchronously or asynchronously
        # For small datasets, run synchronously; for larger ones, run async
        should_run_async = training_status.get("available_training_pairs", 0) > 100
        
        if should_run_async:
            # Start LoRA training in background for large datasets
            import threading
            
            def run_lora_training():
                try:
                    logger.info(f"Starting async LoRA training with min_score={request.min_feedback_score}, max_samples={request.max_samples}")
                    
                    training_result = lora_trainer.train_lora(
                        min_feedback_score=request.min_feedback_score,
                        max_samples=request.max_samples,
                        force_training=request.force_retrain
                    )
                    
                    # Store training completion status
                    memory.write(
                        content=f"LORA_TRAINING_SESSION_COMPLETED: {session_id} - status: {training_result.get('status')}",
                        entry_type="training_session",
                        session_id=session_id,
                        score=5.0 if training_result.get('status') == 'completed' else 0.0
                    )
                    
                    logger.info(f"Async LoRA training session {session_id} completed with status: {training_result.get('status')}")
                    
                except Exception as e:
                    logger.error(f"Async LoRA training session {session_id} failed: {e}")
                    memory.write(
                        content=f"LORA_TRAINING_SESSION_FAILED: {session_id} - error: {str(e)}",
                        entry_type="training_session",
                        session_id=session_id,
                        score=0.0
                    )
            
            # Start training in background thread
            training_thread = threading.Thread(target=run_lora_training)
            training_thread.daemon = True
            training_thread.start()
            
            return {
                "status": "started",
                "session_id": session_id,
                "mode": "lora_async",
                "training_data_available": training_status.get("available_training_pairs", 0),
                "good_feedback_entries": training_status.get("good_feedback_entries", 0),
                "message": f"LoRA training started in background with {training_status.get('available_training_pairs', 0)} training pairs",
                "estimated_duration_minutes": max(5, training_status.get("available_training_pairs", 0) * 0.1),
                "check_status_url": f"/train/status/{session_id}"
            }
        else:
            # Run training synchronously for small datasets
            logger.info(f"Starting synchronous LoRA training with min_score={request.min_feedback_score}, max_samples={request.max_samples}")
            
            training_result = lora_trainer.train_lora(
                min_feedback_score=request.min_feedback_score,
                max_samples=request.max_samples,
                force_training=request.force_retrain
            )
            
            # Store result
            memory.write(
                content=f"LORA_TRAINING_SYNC: status={training_result.get('status')}, examples={training_result.get('training_examples', 0)}",
                entry_type="training_session",
                session_id=session_id,
                score=5.0 if training_result.get('status') == 'completed' else 0.0
            )
            
            return {
                "status": training_result.get('status', 'unknown'),
                "session_id": session_id,
                "mode": "lora_sync", 
                "training_result": training_result,
                "training_status": training_status
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LoRA training session failed: {e}")
        raise HTTPException(status_code=500, detail=f"LoRA training failed: {str(e)}")

@app.get("/train/validate")
def validate_current_adapter():
    """Validate the current LoRA adapter for model collapse and knowledge retention"""
    try:
        if not lora_trainer:
            raise HTTPException(status_code=503, detail="LoRA trainer not available")
        
        logger.info("Starting current adapter validation...")
        
        validation_result = lora_trainer.validate_current_adapter()
        
        if "error" in validation_result:
            return {
                "status": "error",
                "message": validation_result["error"],
                "recommendation": "Check if adapter exists and is properly configured"
            }
        
        # Add actionable recommendations based on validation results
        overall_health = validation_result.get("overall_health", "unknown")
        recommendations = validation_result.get("recommendations", [])
        
        if overall_health == "unhealthy":
            recommendations.extend([
                "Consider reverting to a previous adapter version",
                "Collect more diverse training data",
                "Review feedback quality before next training"
            ])
        elif overall_health == "concerning":
            recommendations.extend([
                "Monitor next few training cycles carefully",
                "Focus on improving training data diversity"
            ])
        
        return {
            "status": "completed",
            "validation_result": validation_result,
            "actionable_recommendations": recommendations,
            "health_summary": {
                "overall_health": overall_health,
                "knowledge_retained": validation_result.get("knowledge_retention", {}).get("passed", False),
                "diversity_score": validation_result.get("diversity_metrics", {}).get("overall_diversity", 0),
                "safe_for_continued_use": overall_health in ["healthy", "acceptable"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Adapter validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.get("/train/health")
def get_training_health():
    """Get comprehensive training system health metrics"""
    try:
        if not lora_trainer:
            return {"error": "LoRA trainer not available"}
        
        # Get training status with health metrics
        status = lora_trainer.get_training_status()
        
        # Get adapter health
        adapter_health = status.get("adapter_health", {})
        data_quality = status.get("data_quality", {})
        
        # Determine overall system health
        system_health = "healthy"
        issues = []
        recommendations = []
        
        # Check adapter health
        if adapter_health.get("status") == "unhealthy":
            system_health = "unhealthy"
            issues.append("Current adapter shows signs of degradation")
            recommendations.append("Validate or retrain current adapter")
        elif adapter_health.get("status") == "concerning":
            system_health = "concerning"
            issues.append("Current adapter may have some issues")
        
        # Check data quality
        if data_quality.get("score", 0) < 0.5:
            if system_health == "healthy":
                system_health = "concerning"
            issues.append("Poor training data quality detected")
            recommendations.extend(data_quality.get("issues", []))
        
        # Check training readiness
        if not status.get("ready_for_training", False):
            recommendations.append("Collect more high-quality feedback")
        
        return {
            "system_health": system_health,
            "training_status": status,
            "issues": issues,
            "recommendations": recommendations,
            "safeguards_active": True,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get training health: {e}")
        return {"error": str(e)}

@app.get("/train/status")
def get_training_status():
    """Get current LoRA training status and readiness"""
    try:
        if not lora_trainer:
            return {"error": "LoRA trainer not available", "ml_available": False}
        
        return lora_trainer.get_training_status()
        
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@app.get("/train/status/{session_id}")
def get_training_session_status(session_id: str):
    """Get status of a specific LoRA training session"""
    try:
        # Check for training session entries in memory
        session_entries = memory.read_by_session(session_id)
        
        if not session_entries:
            raise HTTPException(status_code=404, detail="Training session not found")
        
        # Find the latest training status
        training_entry = None
        for entry in reversed(session_entries):
            if entry.entry_type == "training_session":
                training_entry = entry
                break
        
        if training_entry:
            if "COMPLETED" in training_entry.content:
                return {
                    "status": "completed",
                    "session_id": session_id,
                    "message": "LoRA training completed successfully",
                    "details": training_entry.content,
                    "score": training_entry.score
                }
            elif "FAILED" in training_entry.content:
                return {
                    "status": "failed", 
                    "session_id": session_id,
                    "message": "LoRA training failed",
                    "error": training_entry.content
                }
            elif "SYNC" in training_entry.content:
                return {
                    "status": "completed",
                    "session_id": session_id,
                    "message": "Synchronous LoRA training completed",
                    "details": training_entry.content,
                    "score": training_entry.score
                }
            else:
                return {
                    "status": "in_progress",
                    "session_id": session_id,
                    "message": "LoRA training in progress",
                    "details": training_entry.content
                }
        else:
            return {
                "status": "not_found",
                "session_id": session_id,
                "message": "No training session found with this ID"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training session status: {e}")
        raise HTTPException(status_code=500, detail=f"Session status check failed: {str(e)}")

@app.get("/train/history")
def get_training_history(limit: int = 10):
    """Get history of training sessions"""
    try:
        training_sessions = memory.read_by_type("training_session", limit=limit)
        training_results = memory.read_by_type("training_result", limit=limit)
        
        history = []
        
        for session in training_sessions:
            history.append({
                "session_id": session.session_id,
                "timestamp": session.timestamp.isoformat(),
                "content": session.content,
                "score": session.score,
                "type": "session"
            })
        
        for result in training_results:
            history.append({
                "session_id": result.session_id,
                "timestamp": result.timestamp.isoformat(),
                "content": result.content,
                "score": result.score,
                "type": "result"
            })
        
        # Sort by timestamp
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {"training_history": history[:limit]}
        
    except Exception as e:
        logger.error(f"Failed to get training history: {e}")
        raise HTTPException(status_code=500, detail=f"History retrieval failed: {str(e)}")

@app.post("/generate", response_model=GenerationResponse)
def generate(prompt: Prompt):
    """Generate text from a prompt and automatically score it"""
    try:
        # Generate session ID if not provided
        session_id = prompt.session_id or str(uuid.uuid4())
        
        retrieval = _build_retrieval_context(prompt.text, max_items=3)
        if retrieval["items"]:
            context_str = "\n".join([f"Context snippet: {c}" for c in retrieval["items"]])
            augmented_prompt = (
                "You are an adaptive assistant. Ground answer using retrieved context where helpful; "
                "if irrelevant, proceed normally.\n\nRetrieved Context:\n" + context_str +
                f"\n\nUser Prompt: {prompt.text}\nAnswer:"
            )
        else:
            augmented_prompt = prompt.text

        # Store the prompt
        memory.write(
            content=prompt.text,
            entry_type="prompt",
            session_id=session_id
        )
        
        # Generate output
        if llm:
            output = llm.generate(augmented_prompt)
        else:
            # Mock output for development
            output = f"Mock response to: {prompt.text[:50]}..."
            logger.warning("Using mock LLM output")
        
        # Store the output
        memory.write(
            content=output,
            entry_type="output",
            session_id=session_id
        )
        
        # Get recent memory for scoring context
        recent_memory = memory
        
        # Score the output automatically
        auto_score = critic.score(prompt.text, output, recent_memory)
        
        # Store automatic scoring as internal feedback
        memory.write(
            content=f"AUTO_SCORE: {prompt.text} -> {output} | score={auto_score:.2f}",
            entry_type="internal_feedback",
            session_id=session_id,
            score=auto_score
        )
        
        logger.info(f"Generated response for session {session_id} with auto-score {auto_score:.2f}")
        
        return GenerationResponse(
            output=output,
            session_id=session_id,
            score=auto_score
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/feedback")
def feedback_endpoint(request: FeedbackRequest):
    """Submit feedback for LLM responses (updated endpoint for dashboard)"""
    try:
        # Validate score range (dashboard uses 1-10, convert to 0-5 for internal use)
        if not (1 <= request.score <= 10):
            raise HTTPException(status_code=400, detail="Score must be between 1 and 10")
        
        # Convert 1-10 scale to 0-5 scale for internal consistency
        internal_score = (request.score - 1) * 5 / 9
        
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Store the feedback
        feedback_content = f"DASHBOARD_FEEDBACK: response_id={request.response_id} | score={request.score}/10 (internal={internal_score:.2f})"
        if request.feedback:
            feedback_content += f" | comment={request.feedback}"
            
        memory.write(
            content=feedback_content,
            entry_type="feedback",
            session_id=session_id,
            score=internal_score
        )
        
        logger.info(f"Received dashboard feedback for response {request.response_id} with score {request.score}/10")
        
        # Get feedback count
        feedback_count = len(memory.get_feedback_entries())
        
        return {
            "status": "ok",
            "message": f"Feedback recorded successfully",
            "feedback_id": str(uuid.uuid4()),
            "total_feedback_entries": feedback_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback recording failed: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback recording failed: {str(e)}")

@app.post("/feedback_old")
def feedback(fb: Feedback):
    """Submit human feedback for model improvement"""
    try:
        # Generate session ID if not provided
        session_id = fb.session_id or str(uuid.uuid4())
        
        # Validate score range
        if not (0.0 <= fb.score <= 5.0):
            raise HTTPException(status_code=400, detail="Score must be between 0.0 and 5.0")
        
        # Store the feedback
        feedback_content = f"FEEDBACK: {fb.prompt} -> {fb.output} | score={fb.score}"
        if fb.comment:
            feedback_content += f" | comment={fb.comment}"
            
        memory.write(
            content=feedback_content,
            entry_type="feedback",
            session_id=session_id,
            score=fb.score
        )
        
        logger.info(f"Received feedback for session {session_id} with score {fb.score}")
        
        # Check if we have enough feedback to potentially trigger retraining
        feedback_count = len(memory.get_feedback_entries())
        
        return {
            "status": "ok",
            "message": f"Feedback recorded (total feedback entries: {feedback_count})"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback recording failed: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback recording failed: {str(e)}")

@app.get("/stats")
def get_stats():
    """Get system statistics"""
    try:
        all_entries = memory.read_all()
        feedback_entries = memory.get_feedback_entries()
        
        # Calculate average scores
        scores = [f.score for f in feedback_entries if f.score is not None]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Get recent performance
        recent_feedback = feedback_entries[-20:] if len(feedback_entries) >= 20 else feedback_entries
        recent_scores = [f.score for f in recent_feedback if f.score is not None]
        recent_avg = sum(recent_scores) / len(recent_scores) if recent_scores else 0.0
        
        stats_payload = {
            "total_entries": len(all_entries),
            "feedback_entries": len(feedback_entries),
            "average_score": round(avg_score, 2),
            "recent_average_score": round(recent_avg, 2),
            "entries_by_type": {
                "prompt": len(memory.read_by_type("prompt")),
                "output": len(memory.read_by_type("output")),
                "feedback": len(memory.read_by_type("feedback")),
                "internal_feedback": len(memory.read_by_type("internal_feedback")),
            }
        }
        if metrics:
            stats_payload["metrics"] = metrics.snapshot()
        return stats_payload
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

@app.get("/detailed_score/{session_id}")
def get_detailed_score(session_id: str):
    """Get detailed scoring breakdown for a session"""
    try:
        session_entries = memory.read_by_session(session_id)
        
        if not session_entries:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Find prompt and output from session
        prompt_entry = next((e for e in session_entries if e.entry_type == "prompt"), None)
        output_entry = next((e for e in session_entries if e.entry_type == "output"), None)
        
        if not (prompt_entry and output_entry):
            raise HTTPException(status_code=404, detail="Incomplete session data")
        
        # Get recent memory for context
        recent_memory = memory
        
        # Get detailed scores
        detailed_scores = critic.get_detailed_scores(
            prompt_entry.content, 
            output_entry.content, 
            recent_memory
        )
        
        return {
            "session_id": session_id,
            "prompt": prompt_entry.content,
            "output": output_entry.content,
            "detailed_scores": detailed_scores
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detailed score retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Score retrieval failed: {str(e)}")

@app.post("/trigger_training")
def trigger_training():
    """Manually trigger model retraining (for testing)"""
    try:
        # This would typically be called by a separate service
        # For now, we'll just return information about readiness
        
        feedback_entries = memory.get_feedback_entries()
        training_data = memory.get_training_data()
        
        return {
            "message": "Training trigger received",
            "feedback_entries": len(feedback_entries),
            "training_pairs": len(training_data),
            "ready_for_training": len(training_data) > 10
        }
        
    except Exception as e:
        logger.error(f"Training trigger failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training trigger failed: {str(e)}")

@app.post("/self_learning/start_session")
def start_self_learning_session(iterations: int = 5):
    """Start a self-learning session where the AI generates and evaluates its own responses"""
    try:
        if not self_learner:
            raise HTTPException(status_code=503, detail="Self-learning module not available")
        
        if iterations < 1 or iterations > 20:
            raise HTTPException(status_code=400, detail="Iterations must be between 1 and 20")
        
        logger.info(f"Starting self-learning session with {iterations} iterations")
        results = self_learner.conduct_self_learning_session(iterations)
        
        return {
            "status": "completed",
            "session_id": results["session_id"],
            "iterations_completed": iterations,
            "average_score": round(results["average_score"], 3),
            "best_score": round(results["best_response"]["score"], 3) if results["best_response"] else 0,
            "improvement_trend": [round(trend, 3) for trend in results["improvement_trend"]],
            "areas_for_focus": results["areas_for_focus"],
            "duration_seconds": (results["end_time"] - results["start_time"]).total_seconds()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Self-learning session failed: {e}")
        raise HTTPException(status_code=500, detail=f"Self-learning failed: {str(e)}")

@app.get("/self_learning/insights")
def get_self_learning_insights(days_back: int = 7):
    """Get insights from recent self-learning sessions"""
    try:
        if not self_learner:
            raise HTTPException(status_code=503, detail="Self-learning module not available")
        
        insights = self_learner.get_learning_insights(days_back)
        return insights
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get learning insights: {e}")
        raise HTTPException(status_code=500, detail=f"Insights retrieval failed: {str(e)}")

@app.get("/self_learning/status")
def get_self_learning_status():
    """Get current status of self-learning capabilities"""
    try:
        if not self_learner:
            return {
                "available": False,
                "reason": "Self-learning module not initialized"
            }
        
        # Get some basic stats about self-learning
        recent_entries = memory.read_by_type("self_feedback", limit=10)
        
        return {
            "available": True,
            "recent_sessions": len(set(entry.session_id for entry in recent_entries if entry.session_id and "self_learning" in entry.session_id)),
            "total_self_evaluations": len(recent_entries),
            "knowledge_domains": len(self_learner.knowledge_domains),
            "question_types": len(self_learner.question_types)
        }
        
    except Exception as e:
        logger.error(f"Failed to get self-learning status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")
