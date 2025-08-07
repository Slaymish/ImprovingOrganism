#!/usr/bin/env python3
"""
Continuous Self-Learning Script
Runs periodic self-learning sessions to improve the model autonomously
"""

import time
import logging
import argparse
import schedule
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.self_learning import SelfLearningModule
from src.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/self_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContinuousLearner:
    def __init__(self, iterations_per_session=5, min_interval_hours=2):
        self.self_learner = SelfLearningModule()
        self.iterations_per_session = iterations_per_session
        self.min_interval_hours = min_interval_hours
        self.session_count = 0
        
    def run_learning_session(self):
        """Run a single self-learning session"""
        try:
            logger.info(f"Starting continuous learning session #{self.session_count + 1}")
            
            results = self.self_learner.conduct_self_learning_session(
                num_iterations=self.iterations_per_session
            )
            
            self.session_count += 1
            
            logger.info(f"Session completed successfully:")
            logger.info(f"  - Average Score: {results['average_score']:.3f}")
            logger.info(f"  - Best Score: {results['best_response']['score']:.3f}")
            logger.info(f"  - Areas for Focus: {[area[0] for area in results['areas_for_focus']]}")
            
            # Check if performance is improving
            if results['improvement_trend']:
                avg_improvement = sum(results['improvement_trend']) / len(results['improvement_trend'])
                if avg_improvement > 0.05:
                    logger.info(f"✅ Performance improving (avg trend: +{avg_improvement:.3f})")
                elif avg_improvement < -0.05:
                    logger.warning(f"⚠️ Performance declining (avg trend: {avg_improvement:.3f})")
                else:
                    logger.info(f"📊 Performance stable (avg trend: {avg_improvement:.3f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Learning session failed: {e}", exc_info=True)
            return None

    def get_adaptive_schedule(self):
        """Determine optimal scheduling based on recent performance"""
        try:
            insights = self.self_learner.get_learning_insights(days_back=3)
            
            if "recent_trend" in insights:
                if insights["recent_trend"] == "improving":
                    # More frequent sessions when improving
                    return max(1, self.min_interval_hours - 1)
                elif insights["recent_trend"] == "declining":
                    # Less frequent sessions when declining (might need human intervention)
                    return self.min_interval_hours + 2
                else:
                    return self.min_interval_hours
            
        except Exception as e:
            logger.warning(f"Could not determine adaptive schedule: {e}")
            
        return self.min_interval_hours

    def start_continuous_learning(self):
        """Start the continuous learning loop"""
        logger.info("Starting continuous self-learning system")
        logger.info(f"Configuration:")
        logger.info(f"  - Iterations per session: {self.iterations_per_session}")
        logger.info(f"  - Minimum interval: {self.min_interval_hours} hours")
        
        # Schedule initial session
        schedule.every(self.min_interval_hours).hours.do(self.run_learning_session)
        
        # Run an immediate session on startup
        self.run_learning_session()
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
                # Adaptive scheduling: adjust frequency based on performance
                if self.session_count > 0 and self.session_count % 5 == 0:
                    new_interval = self.get_adaptive_schedule()
                    if new_interval != self.min_interval_hours:
                        logger.info(f"Adjusting schedule interval from {self.min_interval_hours}h to {new_interval}h")
                        schedule.clear()
                        schedule.every(new_interval).hours.do(self.run_learning_session)
                        self.min_interval_hours = new_interval
                
        except KeyboardInterrupt:
            logger.info("Continuous learning stopped by user")
        except Exception as e:
            logger.error(f"Continuous learning failed: {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="Continuous Self-Learning for ImprovingOrganism")
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=5, 
        help="Number of iterations per learning session (1-20)"
    )
    parser.add_argument(
        "--interval", 
        type=int, 
        default=2, 
        help="Minimum hours between learning sessions"
    )
    parser.add_argument(
        "--single", 
        action="store_true", 
        help="Run a single learning session and exit"
    )
    parser.add_argument(
        "--insights", 
        action="store_true", 
        help="Show learning insights and exit"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.iterations < 1 or args.iterations > 20:
        print("Error: iterations must be between 1 and 20")
        return 1
    
    if args.interval < 1:
        print("Error: interval must be at least 1 hour")
        return 1
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    try:
        learner = ContinuousLearner(
            iterations_per_session=args.iterations,
            min_interval_hours=args.interval
        )
        
        if args.insights:
            # Show insights and exit
            insights = learner.self_learner.get_learning_insights()
            print("\n=== Self-Learning Insights ===")
            print(f"Total Sessions: {insights.get('total_sessions', 0)}")
            print(f"Average Score: {insights.get('average_score', 0):.3f}")
            print(f"Recent Trend: {insights.get('recent_trend', 'unknown')}")
            print(f"Score Distribution: {insights.get('score_distribution', {})}")
            if insights.get('common_strengths'):
                print(f"Common Strengths: {[s[0] for s in insights['common_strengths']]}")
            if insights.get('common_weaknesses'):
                print(f"Common Weaknesses: {[w[0] for w in insights['common_weaknesses']]}")
            return 0
        
        elif args.single:
            # Run single session and exit
            results = learner.run_learning_session()
            if results:
                print(f"\nSession completed with average score: {results['average_score']:.3f}")
                return 0
            else:
                print("Session failed")
                return 1
        
        else:
            # Start continuous learning
            learner.start_continuous_learning()
            return 0
            
    except Exception as e:
        logger.error(f"Failed to start continuous learning: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())
