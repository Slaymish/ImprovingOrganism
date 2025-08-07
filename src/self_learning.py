"""
Self-Learning Module for ImprovingOrganism
Generates prompts, evaluates responses, and provides continuous learning
"""

import logging
import random
import json
import re
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests
from .llm_wrapper import LLMWrapper
from .memory_module import MemoryModule
from .config import settings

logger = logging.getLogger(__name__)

class SelfLearningModule:
    def __init__(self):
        self.llm = LLMWrapper()
        self.memory = MemoryModule()
        self.session_id = f"self_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Knowledge domains for generating diverse prompts
        self.knowledge_domains = [
            "mathematics", "science", "history", "literature", "technology",
            "philosophy", "psychology", "economics", "geography", "biology",
            "chemistry", "physics", "computer_science", "linguistics", "art"
        ]
        
        # Question types for varied learning
        self.question_types = [
            "factual", "analytical", "creative", "problem_solving", 
            "comparison", "explanation", "prediction", "classification"
        ]
        
        # Memory management settings
        self.max_prompt_length = 200  # Shorter prompts for memory efficiency
        self.max_response_length = 100  # Shorter responses for memory efficiency

    def check_memory_status(self):
        """Check and log current memory status"""
        try:
            memory_status = self.llm.get_memory_status()
            if memory_status.get("utilization", 0) > 80:
                logger.warning(f"High memory usage: {memory_status.get('utilization', 0):.1f}%")
                self.llm.clear_memory()
            return memory_status
        except Exception as e:
            logger.warning(f"Could not check memory status: {e}")
            return {}

    def generate_self_prompt(self) -> str:
        """Generate a diverse, educational prompt for self-learning (memory-optimized)"""
        domain = random.choice(self.knowledge_domains)
        question_type = random.choice(self.question_types)
        
        # Shorter, more focused prompt templates for better memory usage
        templates = {
            "factual": [
                f"What is a key concept in {domain}?",
                f"Explain a principle in {domain}.",
                f"Define an important term in {domain}."
            ],
            "analytical": [
                f"How do two concepts in {domain} relate?",
                f"What causes X in {domain}?",
                f"Compare two ideas in {domain}."
            ],
            "creative": [
                f"Imagine a future in {domain}.",
                f"Create an analogy for {domain}.",
                f"Design something new in {domain}."
            ],
            "problem_solving": [
                f"How to solve a {domain} problem?",
                f"What steps for {domain} analysis?",
                f"Improve a {domain} process."
            ],
            "comparison": [
                f"How does {domain} affect daily life?",
                f"How has {domain} changed over time?",
                f"What makes {domain} unique?"
            ],
            "explanation": [
                f"Why is X important in {domain}?",
                f"How does Y work in {domain}?",
                f"What makes Z significant in {domain}?"
            ]
        }
        
        prompt_options = templates.get(question_type, templates["factual"])
        base_prompt = random.choice(prompt_options)
        
        # Create a concise, focused prompt (memory-optimized)
        enhanced_prompt = f"Question about {domain}: {base_prompt}"
        
        # Ensure prompt isn't too long
        if len(enhanced_prompt) > self.max_prompt_length:
            enhanced_prompt = enhanced_prompt[:self.max_prompt_length] + "..."
        
        return enhanced_prompt.strip()

    def evaluate_response_empirically(self, prompt: str, response: str) -> Dict:
        """Evaluate the quality and accuracy of a response using multiple criteria"""
        evaluation = {
            "accuracy_score": 0.0,
            "completeness_score": 0.0,
            "clarity_score": 0.0,
            "factual_errors": [],
            "strengths": [],
            "areas_for_improvement": [],
            "overall_score": 0.0
        }
        
        try:
            # 1. Length and structure analysis
            word_count = len(response.split())
            sentence_count = len([s for s in response.split('.') if s.strip()])
            
            # Completeness based on response length and structure
            if word_count >= 100 and sentence_count >= 3:
                evaluation["completeness_score"] = min(1.0, word_count / 200)
            else:
                evaluation["completeness_score"] = 0.3
                evaluation["areas_for_improvement"].append("Response too brief")
            
            # 2. Clarity analysis (basic linguistic features)
            clarity_indicators = [
                "for example" in response.lower(),
                "in other words" in response.lower(),
                "specifically" in response.lower(),
                "this means" in response.lower(),
                any(word in response.lower() for word in ["because", "therefore", "thus", "however"])
            ]
            evaluation["clarity_score"] = sum(clarity_indicators) / len(clarity_indicators)
            
            # 3. Mathematical accuracy check
            math_accuracy = self._check_mathematical_accuracy(response)
            if math_accuracy is not None:
                evaluation["accuracy_score"] = math_accuracy
                if math_accuracy < 0.8:
                    evaluation["factual_errors"].append("Mathematical calculation errors detected")
            else:
                evaluation["accuracy_score"] = 0.7  # Default for non-mathematical content
            
            # 4. Logical structure check
            structure_score = self._evaluate_logical_structure(response)
            evaluation["clarity_score"] = (evaluation["clarity_score"] + structure_score) / 2
            
            # 5. Knowledge consistency check
            consistency_score = self._check_knowledge_consistency(prompt, response)
            evaluation["accuracy_score"] = (evaluation["accuracy_score"] + consistency_score) / 2
            
            # 6. Identify strengths
            if evaluation["completeness_score"] > 0.8:
                evaluation["strengths"].append("Comprehensive response")
            if evaluation["clarity_score"] > 0.7:
                evaluation["strengths"].append("Clear and well-structured")
            if evaluation["accuracy_score"] > 0.8:
                evaluation["strengths"].append("Accurate information")
            
            # Calculate overall score
            evaluation["overall_score"] = (
                evaluation["accuracy_score"] * 0.4 +
                evaluation["completeness_score"] * 0.3 +
                evaluation["clarity_score"] * 0.3
            )
            
        except Exception as e:
            logger.error(f"Error in empirical evaluation: {e}")
            evaluation["overall_score"] = 0.5  # Neutral score on error
            evaluation["areas_for_improvement"].append("Evaluation error occurred")
        
        return evaluation

    def _check_mathematical_accuracy(self, response: str) -> Optional[float]:
        """Check mathematical calculations in the response"""
        # Look for mathematical expressions and verify them
        math_patterns = [
            r'(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)',
            r'(\d+)\s*-\s*(\d+)\s*=\s*(\d+)',
            r'(\d+)\s*\*\s*(\d+)\s*=\s*(\d+)',
            r'(\d+)\s*/\s*(\d+)\s*=\s*(\d+(?:\.\d+)?)',
            r'(\d+)\^(\d+)\s*=\s*(\d+)',
        ]
        
        operations = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b if b != 0 else None,
            '^': lambda a, b: a ** b
        }
        
        total_calculations = 0
        correct_calculations = 0
        
        for i, pattern in enumerate(math_patterns):
            matches = re.findall(pattern, response)
            for match in matches:
                total_calculations += 1
                try:
                    if i == 0:  # Addition
                        expected = int(match[0]) + int(match[1])
                        actual = int(match[2])
                    elif i == 1:  # Subtraction
                        expected = int(match[0]) - int(match[1])
                        actual = int(match[2])
                    elif i == 2:  # Multiplication
                        expected = int(match[0]) * int(match[1])
                        actual = int(match[2])
                    elif i == 3:  # Division
                        expected = int(match[0]) / int(match[1])
                        actual = float(match[2])
                    elif i == 4:  # Exponentiation
                        expected = int(match[0]) ** int(match[1])
                        actual = int(match[2])
                    
                    if abs(expected - actual) < 0.001:  # Allow small floating point errors
                        correct_calculations += 1
                        
                except (ValueError, ZeroDivisionError):
                    continue
        
        if total_calculations == 0:
            return None  # No mathematical calculations found
        
        return correct_calculations / total_calculations

    def _evaluate_logical_structure(self, response: str) -> float:
        """Evaluate the logical structure and flow of the response"""
        score = 0.0
        
        # Check for introduction, body, conclusion structure
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if len(sentences) >= 3:
            score += 0.3
        
        # Check for transitional phrases
        transitions = [
            "first", "second", "third", "finally", "in conclusion",
            "furthermore", "moreover", "additionally", "however",
            "on the other hand", "in contrast", "similarly"
        ]
        
        transition_count = sum(1 for phrase in transitions if phrase in response.lower())
        score += min(0.4, transition_count * 0.1)
        
        # Check for examples or evidence
        evidence_indicators = ["for example", "such as", "including", "specifically"]
        if any(indicator in response.lower() for indicator in evidence_indicators):
            score += 0.3
        
        return min(1.0, score)

    def _check_knowledge_consistency(self, prompt: str, response: str) -> float:
        """Check if the response is consistent with the prompt and general knowledge"""
        # Basic consistency checks
        score = 0.7  # Default baseline
        
        # Check if response addresses the prompt
        prompt_keywords = set(re.findall(r'\b\w+\b', prompt.lower()))
        response_keywords = set(re.findall(r'\b\w+\b', response.lower()))
        
        # Calculate keyword overlap
        overlap = len(prompt_keywords.intersection(response_keywords))
        total_prompt_keywords = len(prompt_keywords)
        
        if total_prompt_keywords > 0:
            keyword_score = min(1.0, overlap / (total_prompt_keywords * 0.3))
            score = (score + keyword_score) / 2
        
        # Check for contradictory statements (basic patterns)
        contradictions = [
            ("always", "never"),
            ("all", "none"),
            ("impossible", "possible"),
            ("true", "false")
        ]
        
        response_lower = response.lower()
        for word1, word2 in contradictions:
            if word1 in response_lower and word2 in response_lower:
                # Check if they're in the same sentence (might indicate contradiction)
                sentences = response_lower.split('.')
                for sentence in sentences:
                    if word1 in sentence and word2 in sentence:
                        score *= 0.8  # Penalize potential contradictions
                        break
        
        return score

    def conduct_self_learning_session(self, num_iterations: int = 5) -> Dict:
        """Conduct a self-learning session with multiple prompt-response cycles"""
        session_results = {
            "session_id": self.session_id,
            "start_time": datetime.now(),
            "iterations": [],
            "average_score": 0.0,
            "improvement_trend": [],
            "best_response": None,
            "areas_for_focus": []
        }
        
        scores = []
        
        for i in range(num_iterations):
            logger.info(f"Starting self-learning iteration {i+1}/{num_iterations}")
            
            try:
                # Check memory status before each iteration
                self.check_memory_status()
                
                # Generate a prompt (shorter for memory efficiency)
                prompt = self.generate_self_prompt()
                
                # Generate response with reduced token limit for memory constraints
                response = self.llm.generate(prompt, max_tokens=self.max_response_length)
                
                # Handle generation errors gracefully
                if "error" in response.lower() or "cuda out of memory" in response.lower():
                    logger.warning(f"Generation issue in iteration {i+1}, using fallback")
                    response = f"Brief response about the topic: {prompt[:50]}... [Memory-limited response]"
                
                # Evaluate the response
                evaluation = self.evaluate_response_empirically(prompt, response)
                
                # Store in memory
                self.memory.store_entry(
                    content=prompt,
                    entry_type="self_prompt",
                    session_id=self.session_id
                )
                
                self.memory.store_entry(
                    content=response,
                    entry_type="self_output",
                    session_id=self.session_id,
                    score=evaluation["overall_score"]
                )
                
                # Store evaluation as feedback (compact format)
                feedback_content = json.dumps({
                    "score": evaluation["overall_score"],
                    "iteration": i + 1,
                    "errors": len(evaluation.get("factual_errors", [])),
                    "strengths": len(evaluation.get("strengths", []))
                })
                
                self.memory.store_entry(
                    content=feedback_content,
                    entry_type="self_feedback",
                    session_id=self.session_id,
                    score=evaluation["overall_score"]
                )
                
                iteration_result = {
                    "iteration": i + 1,
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,  # Truncate for memory
                    "response": response[:200] + "..." if len(response) > 200 else response,  # Truncate for memory
                    "evaluation": evaluation,
                    "score": evaluation["overall_score"]
                }
                
                session_results["iterations"].append(iteration_result)
                scores.append(evaluation["overall_score"])
                
                # Track best response
                if (session_results["best_response"] is None or 
                    evaluation["overall_score"] > session_results["best_response"]["score"]):
                    session_results["best_response"] = iteration_result
                
                logger.info(f"Iteration {i+1} completed with score: {evaluation['overall_score']:.3f}")
                
                # Clear memory after each iteration
                self.llm.clear_memory()
                
            except Exception as e:
                logger.error(f"Error in iteration {i+1}: {e}")
                # Continue with next iteration even if one fails
                scores.append(0.5)  # Neutral score for failed iteration
            finally:
                # Ensure memory is cleared to prevent OOM issues
                self.llm.clear_memory()
                iteration_result = {
                    "iteration": i + 1,
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,  # Truncate for memory
                    "response": response[:200] + "..." if len(response) > 200 else response,  # Truncate for memory
                    "evaluation": evaluation,
                    "score": evaluation["overall_score"]
                }
                session_results["iterations"].append(iteration_result)
                scores.append(evaluation["overall_score"])

            session_results["iterations"].append(iteration_result)
            scores.append(evaluation["overall_score"])
            
            # Track best response
            if (session_results["best_response"] is None or 
                evaluation["overall_score"] > session_results["best_response"]["score"]):
                session_results["best_response"] = iteration_result
            
            logger.info(f"Iteration {i+1} completed with score: {evaluation['overall_score']:.3f}")
        
        # Calculate session statistics
        session_results["average_score"] = sum(scores) / len(scores)
        session_results["end_time"] = datetime.now()
        
        # Calculate improvement trend
        if len(scores) > 1:
            for i in range(1, len(scores)):
                session_results["improvement_trend"].append(scores[i] - scores[i-1])
        
        # Identify areas for focus based on common weaknesses
        all_areas = []
        for iteration in session_results["iterations"]:
            all_areas.extend(iteration["evaluation"]["areas_for_improvement"])
        
        # Count frequency of issues
        area_counts = {}
        for area in all_areas:
            area_counts[area] = area_counts.get(area, 0) + 1
        
        # Sort by frequency and take top issues
        session_results["areas_for_focus"] = sorted(
            area_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        logger.info(f"Self-learning session completed. Average score: {session_results['average_score']:.3f}")
        
        return session_results

    def get_learning_insights(self, days_back: int = 7) -> Dict:
        """Analyze learning progress over time"""
        entries = self.memory.get_entries_by_type("self_feedback", limit=100)
        
        if not entries:
            return {"message": "No self-learning data available"}
        
        scores = []
        timestamps = []
        evaluations = []
        
        for entry in entries:
            try:
                feedback_data = json.loads(entry.content)
                if "evaluation" in feedback_data:
                    scores.append(feedback_data["evaluation"]["overall_score"])
                    timestamps.append(entry.timestamp)
                    evaluations.append(feedback_data["evaluation"])
            except (json.JSONDecodeError, KeyError):
                continue
        
        if not scores:
            return {"message": "No valid evaluation data found"}
        
        # Calculate trends and insights
        insights = {
            "total_sessions": len(scores),
            "average_score": sum(scores) / len(scores),
            "highest_score": max(scores),
            "lowest_score": min(scores),
            "recent_trend": "stable",
            "common_strengths": [],
            "common_weaknesses": [],
            "score_distribution": {
                "excellent": len([s for s in scores if s >= 0.8]),
                "good": len([s for s in scores if 0.6 <= s < 0.8]),
                "fair": len([s for s in scores if 0.4 <= s < 0.6]),
                "poor": len([s for s in scores if s < 0.4])
            }
        }
        
        # Calculate trend
        if len(scores) >= 5:
            recent_scores = scores[-5:]
            earlier_scores = scores[-10:-5] if len(scores) >= 10 else scores[:-5]
            
            if earlier_scores:
                recent_avg = sum(recent_scores) / len(recent_scores)
                earlier_avg = sum(earlier_scores) / len(earlier_scores)
                
                if recent_avg > earlier_avg + 0.05:
                    insights["recent_trend"] = "improving"
                elif recent_avg < earlier_avg - 0.05:
                    insights["recent_trend"] = "declining"
        
        # Aggregate strengths and weaknesses
        all_strengths = []
        all_weaknesses = []
        
        for eval_data in evaluations:
            all_strengths.extend(eval_data.get("strengths", []))
            all_weaknesses.extend(eval_data.get("areas_for_improvement", []))
        
        # Count frequencies
        strength_counts = {}
        weakness_counts = {}
        
        for strength in all_strengths:
            strength_counts[strength] = strength_counts.get(strength, 0) + 1
        
        for weakness in all_weaknesses:
            weakness_counts[weakness] = weakness_counts.get(weakness, 0) + 1
        
        insights["common_strengths"] = sorted(
            strength_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        insights["common_weaknesses"] = sorted(
            weakness_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        return insights
