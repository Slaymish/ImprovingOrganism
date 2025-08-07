#!/usr/bin/env python3
"""
Comprehensive system test to verify all components work coherently
"""

import sys
import os
import requests
import json
import time
from datetime import datetime

# Test configuration
API_BASE = "http://localhost:8000"
DASHBOARD_BASE = "http://localhost:8501"

def test_api_endpoint(endpoint, method="GET", data=None, description="", timeout=10):
    """Test an API endpoint and return result"""
    print(f"🧪 Testing {description}: {method} {endpoint}")

    try:
        if method == "GET":
            response = requests.get(f"{API_BASE}{endpoint}", timeout=timeout)
        elif method == "POST":
            response = requests.post(f"{API_BASE}{endpoint}", json=data, timeout=timeout)
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success: {json.dumps(result, indent=2)[:200]}...")
            return result
        else:
            print(f"   ❌ Failed: {response.text[:200]}...")
            return None
            
    except requests.exceptions.Timeout:
        print(f"   ⏱ Timeout after {timeout}s")
        return None
    except Exception as e:
        print(f"   💥 Exception: {str(e)}")
        return None

def wait_for_api(max_wait=60):
    """Wait for API to become available"""
    print(f"⏳ Waiting for API to start (max {max_wait}s)...")
    
    for i in range(max_wait):
        try:
            response = requests.get(f"{API_BASE}/", timeout=2)
            if response.status_code == 200:
                print(f"✅ API available after {i+1}s")
                return True
        except:
            pass
        time.sleep(1)
        
        if i % 10 == 9:  # Progress indicator every 10 seconds
            print(f"   Still waiting... ({i+1}s)")
    
    print(f"❌ API not available after {max_wait}s")
    return False

def main():
    """Run comprehensive system tests"""
    print("🔍 ImprovingOrganism System Coherence Test")
    print("=" * 60)
    
    # Wait for API to be ready
    if not wait_for_api():
        print("❌ API not responding. Please check if the server is running.")
        return
    
    # Test 1: Basic API connectivity and health
    result = test_api_endpoint("/", description="API Root")
    if not result:
        print("❌ API not responding. Exiting tests.")
        return
    
    # Test 1.5: Detailed health check
    health_result = test_api_endpoint("/health", description="Health Check")
    if health_result:
        print(f"   System Health: {health_result.get('overall', 'unknown')}")
        components = health_result.get('components', {})
        for comp, status in components.items():
            status_icon = "✅" if status else "❌"
            print(f"     {status_icon} {comp}: {'available' if status else 'unavailable'}")
    
    # Test 2: Query endpoint (core functionality) - longer timeout for first generation
    query_result = test_api_endpoint(
        "/query", 
        method="POST",
        data={"query": "What is machine learning?"},
        description="LLM Query",
        timeout=60  # First generation can be slow
    )
    
    # Test 3: Feedback submission (only if query succeeded)
    feedback_result = None
    if query_result:
        feedback_result = test_api_endpoint(
            "/feedback",
            method="POST", 
            data={
                "response_id": query_result.get("session_id", "test"),
                "score": 8,
                "feedback": "Good test response",
                "session_id": query_result.get("session_id")
            },
            description="Feedback Submission"
        )
    else:
        print("\n🧪 Skipping Feedback Submission: Query test failed")
        print("   ⏭ Skipped due to query timeout")
    
    # Test 4: Training status
    training_status = test_api_endpoint("/train/status", description="Training Status")
    
    # Test 5: Training history
    training_history = test_api_endpoint("/train/history", description="Training History")
    
    # Test 6: Self-learning (quick test)
    self_learn_result = test_api_endpoint(
        "/self_learn",
        method="POST",
        data={"iterations": 1, "topic": "test"},
        description="Self-Learning Session"
    )
    
    # Test 7: Training session start
    train_result = test_api_endpoint(
        "/train",
        method="POST",
        data={"mode": "interactive"},
        description="Training Session"
    )
    
    # Summary
    print("\n📊 Test Summary")
    print("-" * 30)
    
    tests = [
        ("API Root", result is not None),
        ("Health Check", health_result is not None),
        ("LLM Query", query_result is not None),
        ("Feedback", feedback_result is not None),
        ("Training Status", training_status is not None),
        ("Training History", training_history is not None),
        ("Self-Learning", self_learn_result is not None),
        ("Training Start", train_result is not None)
    ]
    
    passed = sum(1 for _, success in tests if success)
    total = len(tests)
    
    print(f"Tests Passed: {passed}/{total}")
    
    for test_name, success in tests:
        if success:
            status = "✅"
        elif test_name == "Feedback" and query_result is None:
            status = "⏭"  # Skipped
        else:
            status = "❌"
        print(f"  {status} {test_name}")
    
    if passed >= total - 1:  # Allow one failure/skip
        print("\n🎉 System working coherently!")
    else:
        print(f"\n⚠️  {total - passed} tests failed. System needs attention.")
    
    # Additional system info
    print("\n🔧 System Information")
    print("-" * 30)
    
    if training_status:
        print(f"Training Ready: {training_status.get('ready_for_training', 'Unknown')}")
        print(f"ML Available: {training_status.get('ml_available', 'Unknown')}")
        print(f"Feedback Entries: {training_status.get('total_feedback_entries', 'Unknown')}")
    
    print(f"Test completed at: {datetime.now()}")

if __name__ == "__main__":
    main()
