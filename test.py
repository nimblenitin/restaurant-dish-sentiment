"""
Test Script for Restaurant Food Recommendation System
Verifies that all components are working correctly
"""

import os
import sys


def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import transformers
        print(f"  ✓ transformers {transformers.__version__}")
    except ImportError as e:
        print(f"  ✗ transformers: {e}")
        return False
    
    try:
        import datasets
        print(f"  ✓ datasets {datasets.__version__}")
    except ImportError as e:
        print(f"  ✗ datasets: {e}")
        return False
    
    try:
        import evaluate
        print(f"  ✓ evaluate")
    except ImportError as e:
        print(f"  ✗ evaluate: {e}")
        return False
    
    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"  ✓ pandas {pd.__version__}")
    except ImportError as e:
        print(f"  ✗ pandas: {e}")
        return False
    
    try:
        import numpy as np
        print(f"  ✓ numpy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False
    
    return True


def test_sample_data():
    """Test that sample data can be created"""
    print("\nTesting sample data creation...")
    
    try:
        from src.scraper import create_sample_data
        df = create_sample_data()
        print(f"  ✓ Created {len(df)} sample reviews")
        return True
    except Exception as e:
        print(f"  ✗ Error creating sample data: {e}")
        return False


def test_label_generation():
    """Test that training labels can be generated"""
    print("\nTesting label generation...")
    
    try:
        import pandas as pd
        from src.label_generator import generate_training_data
        
        # Load sample data
        df = pd.read_csv("data/raw_reviews/all_restaurants.csv")
        
        # Generate training data
        training_data = generate_training_data(df, reviews_per_example=2, max_examples=10)
        print(f"  ✓ Generated {len(training_data)} training examples")
        
        # Show sample
        if training_data:
            print(f"  Sample input: {training_data[0]['input'][:100]}...")
            print(f"  Sample output: {training_data[0]['output'][:100]}...")
        
        return True
    except Exception as e:
        print(f"  ✗ Error generating labels: {e}")
        return False


def test_model_loading():
    """Test that mT5 model can be loaded"""
    print("\nTesting model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        model_checkpoint = "google/mt5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        
        print(f"  ✓ Model loaded: {model.num_parameters():,} parameters")
        print(f"  ✓ Tokenizer vocab size: {tokenizer.vocab_size}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("RUNNING TESTS")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Sample Data", test_sample_data),
        ("Label Generation", test_label_generation),
        ("Model Loading", test_model_loading),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} - {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
