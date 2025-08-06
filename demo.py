#!/usr/bin/env python3
"""
End-to-end demonstration of the ImprovingOrganism system.
This script demonstrates: generate → rate → retrain → regenerate
"""

import time
import requests
import json
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemDemo:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def test_connection(self) -> bool:
        """Test if the API is accessible"""
        try:
            response = requests.get(f"{self.base_url}/")
            return response.status_code == 200
        except:
            return False
    
    def generate_text(self, prompt: str, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Generate text from a prompt"""
        data = {"text": prompt}
        if session_id:
            data["session_id"] = session_id
            
        response = requests.post(f"{self.base_url}/generate", json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Generation failed: {response.text}")
            return None
    
    def submit_feedback(self, prompt: str, output: str, score: float, comment: Optional[str] = None, session_id: Optional[str] = None) -> bool:
        """Submit feedback for generated content"""
        data = {
            "prompt": prompt,
            "output": output,
            "score": score,
            "comment": comment,
            "session_id": session_id
        }
        
        response = requests.post(f"{self.base_url}/feedback", json=data)
        
        if response.status_code == 200:
            logger.info(f"Feedback submitted: {score}/5.0")
            return True
        else:
            logger.error(f"Feedback failed: {response.text}")
            return False
    
    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get system statistics"""
        response = requests.get(f"{self.base_url}/stats")
        if response.status_code == 200:
            return response.json()
        return None
    
    def get_detailed_score(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed scoring breakdown"""
        response = requests.get(f"{self.base_url}/detailed_score/{session_id}")
        if response.status_code == 200:
            return response.json()
        return None
    
    def trigger_training(self) -> Optional[Dict[str, Any]]:
        """Trigger model retraining"""
        response = requests.post(f"{self.base_url}/trigger_training")
        if response.status_code == 200:
            return response.json()
        return None
    
    def run_demo_cycle(self):
        """Run a complete demonstration cycle"""
        
        print("=" * 60)
        print("ImprovingOrganism End-to-End Demonstration")
        print("=" * 60)
        
        # Test connection
        print("\n1. Testing API connection...")
        if not self.test_connection():
            print("❌ API not accessible. Please start the server first.")
            return
        print("✅ API connection successful")
        
        # Initial stats
        print("\n2. Getting initial system stats...")
        initial_stats = self.get_stats()
        if initial_stats:
            print(f"   Total entries: {initial_stats['total_entries']}")
            print(f"   Feedback entries: {initial_stats['feedback_entries']}")
            print(f"   Average score: {initial_stats['average_score']}")
        
        # Generate some content
        print("\n3. Generating initial content...")
        
        test_prompts = [
            "Explain the concept of machine learning",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis",
            "How do neural networks work?",
            "What is the importance of data privacy?"
        ]
        
        generations = []
        
        for i, prompt in enumerate(test_prompts):
            print(f"   Generating response {i+1}/5: {prompt[:30]}...")
            
            result = self.generate_text(prompt)
            if result:
                generations.append({
                    'prompt': prompt,
                    'output': result['output'],
                    'session_id': result['session_id'],
                    'auto_score': result.get('score', 0)
                })
                print(f"   Auto-score: {result.get('score', 0):.2f}/5.0")
            
            time.sleep(1)  # Rate limiting
        
        # Submit varied feedback
        print("\n4. Submitting human feedback...")
        
        feedback_patterns = [
            (4.5, "Excellent explanation with good detail"),
            (3.0, "Adequate but could be more comprehensive"),
            (4.8, "Very clear and well-structured"),
            (2.5, "Too brief and lacks examples"),
            (4.2, "Good coverage of the topic")
        ]
        
        for i, generation in enumerate(generations):
            score, comment = feedback_patterns[i]
            print(f"   Submitting feedback {i+1}/5: {score}/5.0")
            
            success = self.submit_feedback(
                generation['prompt'],
                generation['output'],
                score,
                comment,
                generation['session_id']
            )
            
            if not success:
                print(f"   ❌ Failed to submit feedback {i+1}")
            
            time.sleep(0.5)
        
        # Show detailed scoring for one example
        print("\n5. Getting detailed scoring breakdown...")
        if generations:
            example = generations[0]
            detailed = self.get_detailed_score(example['session_id'])
            if detailed:
                print(f"   Session: {example['session_id']}")
                print(f"   Prompt: {detailed['prompt'][:50]}...")
                print("   Detailed scores:")
                for metric, score in detailed['detailed_scores'].items():
                    print(f"     {metric}: {score:.2f}")
        
        # Updated stats
        print("\n6. Checking updated system stats...")
        updated_stats = self.get_stats()
        if updated_stats and initial_stats:
            print(f"   Total entries: {updated_stats['total_entries']} (was {initial_stats['total_entries']})")
            print(f"   Feedback entries: {updated_stats['feedback_entries']} (was {initial_stats['feedback_entries']})")
            print(f"   Average score: {updated_stats['average_score']:.2f} (was {initial_stats['average_score']:.2f})")
            print("   Entries by type:")
            for entry_type, count in updated_stats['entries_by_type'].items():
                print(f"     {entry_type}: {count}")
        
        # Check training readiness
        print("\n7. Checking training readiness...")
        training_info = self.trigger_training()
        if training_info:
            print(f"   Feedback entries: {training_info['feedback_entries']}")
            print(f"   Training pairs: {training_info['training_pairs']}")
            print(f"   Ready for training: {training_info['ready_for_training']}")
            
            if training_info['ready_for_training']:
                print("   ✅ System has enough data for retraining!")
            else:
                print("   ⏳ More feedback needed before retraining")
        
        # Generate again to show potential improvement
        print("\n8. Generating content again to demonstrate system evolution...")
        
        follow_up_prompt = "Explain machine learning in simple terms"
        print(f"   Prompt: {follow_up_prompt}")
        
        result = self.generate_text(follow_up_prompt)
        if result:
            print(f"   Output: {result['output'][:100]}...")
            print(f"   Auto-score: {result.get('score', 0):.2f}/5.0")
            
            # Get detailed scoring
            detailed = self.get_detailed_score(result['session_id'])
            if detailed:
                print("   Detailed scoring:")
                for metric, score in detailed['detailed_scores'].items():
                    print(f"     {metric}: {score:.2f}")
        
        print("\n" + "=" * 60)
        print("Demonstration complete! 🎉")
        print("The system has now:")
        print("1. ✅ Generated multiple responses")
        print("2. ✅ Collected and stored feedback")
        print("3. ✅ Analyzed content with multiple metrics")
        print("4. ✅ Prepared data for potential retraining")
        print("5. ✅ Demonstrated the full feedback loop")
        print("=" * 60)

def main():
    demo = SystemDemo()
    
    print("Starting ImprovingOrganism demonstration...")
    print("Make sure the API server is running on localhost:8000")
    print("(Run: uvicorn src.main:app --host 0.0.0.0 --port 8000)")
    
    input("\nPress Enter to continue...")
    
    try:
        demo.run_demo_cycle()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        logger.error(f"Demo error: {e}")

if __name__ == "__main__":
    main()
