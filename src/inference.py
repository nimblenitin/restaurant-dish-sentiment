"""
Inference for Restaurant Food Recommendation Model
Use the mT5 model to analyze reviews and recommend dishes
"""

import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List, Dict, Optional


class FoodRecommender:
    """
    Food Recommendation System using mT5
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the food recommender
        
        Args:
            model_path: Path to the fine-tuned model (if None, uses base mT5)
        """
        if model_path and os.path.exists(model_path):
            print(f"\nLoading fine-tuned model from {model_path}...")
            self.model_name = model_path
        else:
            print(f"\nLoading base mT5 model...")
            self.model_name = "google/mt5-small"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def analyze_reviews(self, 
                       reviews: List[str], 
                       restaurant_name: str,
                       max_length: int = 128) -> str:
        """
        Analyze reviews and recommend dishes
        
        Args:
            reviews: List of review texts
            restaurant_name: Name of the restaurant
            max_length: Maximum length of generated output
            
        Returns:
            Food recommendation string
        """
        # Combine reviews into single text
        combined_reviews = " ".join(reviews)
        
        # Create input prompt
        input_text = f"analyze_food: Reviews from {restaurant_name}: {combined_reviews}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(self.device)
        
        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        # Decode output
        recommendation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return recommendation
    
    def analyze_single_review(self, 
                            review: str, 
                            restaurant_name: str,
                            max_length: int = 128) -> str:
        """
        Analyze a single review
        
        Args:
            review: Review text
            restaurant_name: Name of the restaurant
            max_length: Maximum length of generated output
            
        Returns:
            Food recommendation string
        """
        return self.analyze_reviews([review], restaurant_name, max_length)
    
    def analyze_from_csv(self, 
                        csv_path: str, 
                        restaurant_name: str,
                        max_reviews: int = 10) -> str:
        """
        Analyze reviews from a CSV file
        
        Args:
            csv_path: Path to CSV file with reviews
            restaurant_name: Name of the restaurant
            max_reviews: Maximum number of reviews to use
            
        Returns:
            Food recommendation string
        """
        import pandas as pd
        
        # Load reviews
        df = pd.read_csv(csv_path)
        
        # Filter by restaurant if column exists
        if 'restaurant_name' in df.columns:
            df = df[df['restaurant_name'] == restaurant_name]
        
        # Get review texts
        if 'review_body' in df.columns:
            reviews = df['review_body'].tolist()[:max_reviews]
        elif 'text' in df.columns:
            reviews = df['text'].tolist()[:max_reviews]
        else:
            raise ValueError("CSV must contain 'review_body' or 'text' column")
        
        return self.analyze_reviews(reviews, restaurant_name)


def format_output(recommendation: str) -> str:
    """
    Format the recommendation output for better readability
    
    Args:
        recommendation: Raw recommendation string
        
    Returns:
        Formatted string
    """
    # Add some formatting
    formatted = recommendation.replace(". Top 3", "\n\nTop 3")
    formatted = formatted.replace(". ", ".\n")
    
    return formatted


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with mT5 model")
    parser.add_argument("--model", default=None,
                       help="Path to fine-tuned model (optional)")
    parser.add_argument("--restaurant", default="Bar-Bill Tavern",
                       help="Restaurant name")
    parser.add_argument("--reviews", nargs="+",
                       help="List of review texts")
    parser.add_argument("--csv", help="Path to CSV file with reviews")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Initialize recommender
    try:
        recommender = FoodRecommender(args.model)
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("Please make sure you have installed all dependencies!")
        exit(1)
    
    if args.interactive:
        # Interactive mode
        print("\n" + "="*60)
        print("RESTAURANT FOOD RECOMMENDATION SYSTEM")
        print("="*60)
        print("Enter reviews to get food recommendations.")
        print("Type 'quit' to exit.\n")
        
        while True:
            restaurant_name = input("Restaurant name (or 'quit'): ").strip()
            if restaurant_name.lower() == 'quit':
                break
            
            print("Enter reviews (one per line, empty line to finish):")
            reviews = []
            while True:
                review = input("> ").strip()
                if not review:
                    break
                reviews.append(review)
            
            if reviews:
                recommendation = recommender.analyze_reviews(reviews, restaurant_name)
                print("\n" + "="*60)
                print("RECOMMENDATION:")
                print("="*60)
                print(format_output(recommendation))
                print("="*60 + "\n")
    
    elif args.csv:
        # CSV mode
        recommendation = recommender.analyze_from_csv(args.csv, args.restaurant)
        print("\n" + "="*60)
        print("RECOMMENDATION:")
        print("="*60)
        print(format_output(recommendation))
    
    elif args.reviews:
        # Direct reviews mode
        recommendation = recommender.analyze_reviews(args.reviews, args.restaurant)
        print("\n" + "="*60)
        print("RECOMMENDATION:")
        print("="*60)
        print(format_output(recommendation))
    
    else:
        # Demo mode with sample reviews
        print("\n" + "="*60)
        print("DEMO MODE - Using sample reviews")
        print("="*60)
        
        sample_reviews = [
            "The wings here are incredible! Best I've ever had in Buffalo. The honey butter barbecue flavor is amazing.",
            "Beef on weck was perfectly done. The roast beef was tender and the roll was fresh. Highly recommend!",
            "The salad was fresh but overpriced for what you get. Wings are the real star here.",
            "Wings were okay but nothing special. The fries were soggy and cold. Disappointing visit.",
            "Great spot! Wings are crispy and the sauce is perfect. Will definitely come back."
        ]
        
        recommendation = recommender.analyze_reviews(sample_reviews, "Bar-Bill Tavern")
        
        print("\nSample Reviews:")
        for i, review in enumerate(sample_reviews, 1):
            print(f"{i}. {review}")
        
        print("\n" + "="*60)
        print("RECOMMENDATION:")
        print("="*60)
        print(format_output(recommendation))
