from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Union
import uuid
import logging
from .llm_wrapper import LLMWrapper
from .memory_module import MemoryModule
from .critic_module import CriticModule
from .self_learning import SelfLearningModule

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ImprovingOrganism API", version="1.0.0")

# Initialize components
try:
    llm = LLMWrapper()
    memory = MemoryModule()
    critic = CriticModule()
    self_learner = SelfLearningModule()
    logger.info("All components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    # Create mock components for development
    llm = None
    memory = MemoryModule()
    critic = CriticModule()
    self_learner = None

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
    mode: str = "interactive"
    session_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    response_id: str
    score: float  # Will accept int or float, as pydantic auto-converts
    feedback: Optional[str] = None
    session_id: Optional[str] = None

@app.get("/")
def root():
    return {"message": "ImprovingOrganism API", "status": "running"}

@app.post("/query")
def query(request: QueryRequest):
    """Handle LLM queries from the dashboard"""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Store the prompt
        memory.write(
            content=request.query,
            entry_type="prompt",
            session_id=session_id
        )
        
        # Generate output
        if llm:
            output = llm.generate(request.query)
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
        recent_memory = memory.read_all()[-50:]  # Last 50 entries
        
        # Score the output automatically
        auto_score = critic.score(request.query, output, recent_memory)
        
        # Store automatic scoring as internal feedback
        memory.write(
            content=f"AUTO_SCORE: {request.query} -> {output} | score={auto_score:.2f}",
            entry_type="internal_feedback",
            session_id=session_id,
            score=auto_score
        )
        
        logger.info(f"Generated response for session {session_id} with auto-score {auto_score:.2f}")
        
        return {
            "response": output,
            "session_id": session_id,
            "metadata": {
                "auto_score": auto_score,
                "model_name": getattr(llm, 'model_name', 'mock') if llm else 'mock',
                "model_path": getattr(llm, 'model_path', 'none') if llm else 'none',
                "timestamp": memory.read_by_session(session_id)[-1].timestamp if memory.read_by_session(session_id) else None
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
    """Start a training session"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        logger.info(f"Starting training session {session_id} in {request.mode} mode")
        
        # Get training data from memory
        feedback_entries = memory.get_feedback_entries()
        training_data = memory.get_training_data()
        
        # Store training session start
        memory.write(
            content=f"TRAINING_SESSION_START: mode={request.mode}",
            entry_type="training",
            session_id=session_id
        )
        
        # For now, simulate training preparation
        # In a real implementation, this would trigger actual model training
        return {
            "status": "Active",
            "session_id": session_id,
            "mode": request.mode,
            "training_data_available": len(training_data),
            "feedback_entries": len(feedback_entries),
            "message": f"Training session initiated with {len(training_data)} training pairs"
        }
        
    except Exception as e:
        logger.error(f"Training session failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/generate", response_model=GenerationResponse)
def generate(prompt: Prompt):
    """Generate text from a prompt and automatically score it"""
    try:
        # Generate session ID if not provided
        session_id = prompt.session_id or str(uuid.uuid4())
        
        # Store the prompt
        memory.write(
            content=prompt.text,
            entry_type="prompt",
            session_id=session_id
        )
        
        # Generate output
        if llm:
            output = llm.generate(prompt.text)
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
        recent_memory = memory.read_all()[-50:]  # Last 50 entries
        
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
        
        return {
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
        recent_memory = memory.read_all()[-50:]
        
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
