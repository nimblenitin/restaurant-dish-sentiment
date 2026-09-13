"""
Main Pipeline Script for Restaurant Food Recommendation System
Orchestrates the entire workflow from data collection to inference
"""

import os
import sys
import argparse


def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'transformers', 'datasets', 'evaluate', 'rouge_score',
        'torch', 'pandas', 'numpy', 'nltk'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    return True


def run_pipeline(sample_data: bool = True):
    """
    Run the complete pipeline
    
    Args:
        sample_data: If True, use sample data instead of scraping
    """
    print("\n" + "="*60)
    print("RESTAURANT FOOD RECOMMENDATION SYSTEM")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Step 1: Get reviews
    print("\n[Step 1/4] Getting reviews...")
    if sample_data:
        os.system("python src/scraper.py --sample")
    else:
        os.system("python src/scraper.py --limit 500")
    
    # Step 2: Generate training data
    print("\n[Step 2/4] Generating training data...")
    os.system("python src/label_generator.py")
    
    # Step 3: Train model
    print("\n[Step 3/4] Training model...")
    os.system("python src/train.py --epochs 10")
    
    # Step 4: Run inference
    print("\n[Step 4/4] Running inference demo...")
    os.system("python src/inference.py")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print("\nYour model is ready to use!")
    print("Run: python src/inference.py --interactive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete pipeline")
    parser.add_argument("--scrape", action="store_true",
                       help="Scrape actual reviews (requires Google account)")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training and just run inference")
    
    args = parser.parse_args()
    
    if args.skip_training:
        print("\nSkipping training, running inference only...")
        os.system("python src/inference.py")
    else:
        run_pipeline(sample_data=not args.scrape)
