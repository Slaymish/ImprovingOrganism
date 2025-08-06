import re
import json
from typing import List, Dict, Any
from collections import Counter
import math

class CriticModule:
    def __init__(self):
        self.common_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
    def score(self, prompt: str, output: str, memory: List[Any]) -> float:
        """
        Comprehensive scoring that considers:
        - Coherence (grammatical structure, logical flow)
        - Novelty (uniqueness compared to past outputs)
        - Memory alignment (consistency with stored knowledge)
        - Relevance to prompt
        """
        coherence_score = self._score_coherence(output)
        novelty_score = self._score_novelty(output, memory)
        alignment_score = self._score_memory_alignment(prompt, output, memory)
        relevance_score = self._score_relevance(prompt, output)
        
        # Weighted combination
        final_score = (
            0.3 * coherence_score +
            0.25 * novelty_score +
            0.25 * alignment_score +
            0.2 * relevance_score
        )
        
        return min(5.0, max(0.0, final_score))
    
    def _score_coherence(self, text: str) -> float:
        """Score based on grammatical structure and readability"""
        if not text.strip():
            return 0.0
            
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return 1.0
            
        score = 3.0  # Base score
        
        # Check for complete sentences
        complete_sentences = sum(1 for s in sentences if len(s.split()) > 2)
        sentence_ratio = complete_sentences / len(sentences) if sentences else 0
        score += sentence_ratio * 1.0
        
        # Check for punctuation variety
        has_punctuation = any(p in text for p in '.,!?;:')
        if has_punctuation:
            score += 0.5
            
        # Penalize repetitive patterns
        words = text.lower().split()
        if len(words) > 3:
            word_freq = Counter(words)
            most_common_freq = word_freq.most_common(1)[0][1] if word_freq else 1
            if most_common_freq > len(words) * 0.3:  # Too repetitive
                score -= 1.0
                
        # Check for reasonable length
        if 10 <= len(words) <= 200:
            score += 0.5
            
        return score
    
    def _score_novelty(self, output: str, memory: List[Any]) -> float:
        """Score based on uniqueness compared to previous outputs"""
        if not memory:
            return 4.0  # High novelty if no history
            
        output_words = set(output.lower().split()) - self.common_words
        if not output_words:
            return 2.0
            
        similarities = []
        for entry in memory[-20:]:  # Check last 20 entries
            if hasattr(entry, 'content'):
                past_words = set(entry.content.lower().split()) - self.common_words
                if past_words:
                    intersection = len(output_words & past_words)
                    union = len(output_words | past_words)
                    similarity = intersection / union if union > 0 else 0
                    similarities.append(similarity)
        
        if not similarities:
            return 4.0
            
        avg_similarity = sum(similarities) / len(similarities)
        novelty_score = 5.0 * (1 - avg_similarity)  # Higher score for lower similarity
        
        return max(0.0, novelty_score)
    
    def _score_memory_alignment(self, prompt: str, output: str, memory: List[Any]) -> float:
        """Score based on consistency with stored knowledge"""
        if not memory:
            return 3.0  # Neutral if no memory
            
        prompt_words = set(prompt.lower().split())
        output_words = set(output.lower().split())
        
        relevant_memories = []
        for entry in memory:
            if hasattr(entry, 'content'):
                entry_words = set(entry.content.lower().split())
                if len(prompt_words & entry_words) > 0:  # Related to current prompt
                    relevant_memories.append(entry)
        
        if not relevant_memories:
            return 3.0
            
        # Check consistency with relevant memories
        consistency_scores = []
        for memory_entry in relevant_memories[-10:]:  # Recent relevant memories
            memory_words = set(memory_entry.content.lower().split())
            consistency = len(output_words & memory_words) / len(output_words | memory_words) if (output_words | memory_words) else 0
            consistency_scores.append(consistency)
        
        if consistency_scores:
            avg_consistency = sum(consistency_scores) / len(consistency_scores)
            return 2.0 + (3.0 * avg_consistency)  # Scale to 2-5 range
        
        return 3.0
    
    def _score_relevance(self, prompt: str, output: str) -> float:
        """Score how well the output addresses the prompt"""
        if not prompt.strip() or not output.strip():
            return 1.0
            
        prompt_words = set(prompt.lower().split()) - self.common_words
        output_words = set(output.lower().split()) - self.common_words
        
        if not prompt_words:
            return 3.0
            
        # Check word overlap
        overlap = len(prompt_words & output_words)
        overlap_ratio = overlap / len(prompt_words)
        
        base_score = 2.0 + (3.0 * overlap_ratio)
        
        # Bonus for addressing question words
        question_words = {'what', 'when', 'where', 'why', 'how', 'who', 'which'}
        prompt_questions = prompt_words & question_words
        
        if prompt_questions:
            # Check if output seems to address the question type
            if 'what' in prompt_questions and any(word in output.lower() for word in ['is', 'are', 'means', 'refers']):
                base_score += 0.5
            elif 'when' in prompt_questions and any(word in output.lower() for word in ['time', 'date', 'year', 'day']):
                base_score += 0.5
            elif 'where' in prompt_questions and any(word in output.lower() for word in ['location', 'place', 'here', 'there']):
                base_score += 0.5
            elif 'why' in prompt_questions and any(word in output.lower() for word in ['because', 'reason', 'cause', 'since']):
                base_score += 0.5
            elif 'how' in prompt_questions and any(word in output.lower() for word in ['by', 'through', 'method', 'way']):
                base_score += 0.5
        
        return min(5.0, base_score)
    
    def get_detailed_scores(self, prompt: str, output: str, memory: List[Any]) -> Dict[str, float]:
        """Get breakdown of all scoring components"""
        return {
            'coherence': self._score_coherence(output),
            'novelty': self._score_novelty(output, memory),
            'memory_alignment': self._score_memory_alignment(prompt, output, memory),
            'relevance': self._score_relevance(prompt, output),
            'overall': self.score(prompt, output, memory)
        }
