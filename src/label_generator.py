"""
Training Data Generator for Restaurant Food Recommendation Model
Generates input-output pairs for fine-tuning mT5
"""

import re
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import json


# Common food-related keywords to help identify dish mentions
FOOD_KEYWORDS = [
    # Wings and chicken
    "wings", "chicken wings", "buffalo wings", "hot wings", "mild wings",
    "boneless wings", "chicken tenders", "chicken strips", "fried chicken",
    
    # Beef dishes
    "beef on weck", "roast beef", "steak", "burger", "hamburger",
    "beef wellington", "filet", "prime rib", "brisket",
    
    # Pizza
    "pizza", "pepperoni pizza", "cheese pizza", "buffalo pizza",
    "neapolitan pizza", "sicilian pizza",
    
    # Pasta and Italian
    "pasta", "spaghetti", "lasagna", "ravioli", "gnocchi",
    "fettuccine", "penne", "mac and cheese", "alfredo",
    "bolognese", "carbonara", "marinara",
    
    # Seafood
    "fish", "fish fry", "salmon", "shrimp", "lobster", "crab",
    "calamari", "octopus", "tuna", "cod", "haddock",
    
    # Salads
    "salad", "caesar salad", "greek salad", "house salad",
    "garden salad", "wedge salad",
    
    # Soups
    "soup", "chili", "clam chowder", "buffalo chicken soup",
    
    # Appetizers
    "fries", "french fries", "onion rings", "mozzarella sticks",
    "nachos", "wings", "jalapeno poppers", "loaded fries",
    
    # Desserts
    "dessert", "cake", "pie", "ice cream", "cheesecake",
    "tiramisu", "chocolate", "brownie", "cookie",
    
    # Drinks
    "beer", "wine", "cocktail", "margarita", "soda", "coffee",
    
    # Other
    "sandwich", "wrap", "taco", "burrito", "sushi", "ramen"
]


def extract_food_mentions(text: str) -> List[str]:
    """
    Extract food/dish mentions from review text
    
    Args:
        text: Review text
        
    Returns:
        List of food mentions found
    """
    text_lower = text.lower()
    found_foods = []
    
    for food in FOOD_KEYWORDS:
        if food.lower() in text_lower:
            found_foods.append(food)
    
    # Also try to find capitalized food words (proper nouns for dishes)
    words = text.split()
    for i, word in enumerate(words):
        # Check if word looks like a dish name (capitalized, not at start)
        if i > 0 and word[0].isupper() and len(word) > 2:
            # Check if it's likely a food item
            if any(food_word in word.lower() for food_word in ["wing", "steak", "pasta", "pizza", "salad", "soup", "fish", "cake"]):
                if word.lower() not in [f.lower() for f in found_foods]:
                    found_foods.append(word)
    
    return list(set(found_foods))


def determine_sentiment(review_text: str, rating: Optional[int] = None) -> str:
    """
    Determine sentiment of a review based on text and rating
    
    Args:
        review_text: Review text
        rating: Star rating (1-5) if available
        
    Returns:
        Sentiment: "positive", "negative", or "neutral"
    """
    # Use rating if available
    if rating is not None:
        if rating >= 4:
            return "positive"
        elif rating <= 2:
            return "negative"
        else:
            return "neutral"
    
    # Simple text-based sentiment analysis
    positive_words = ["amazing", "excellent", "great", "best", "love", "perfect", "delicious", "fantastic", "wonderful", "incredible"]
    negative_words = ["terrible", "awful", "bad", "worst", "hate", "disappointing", "overpriced", "cold", "soggy", "dry"]
    
    text_lower = review_text.lower()
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"


def generate_training_example(reviews: List[Dict], restaurant_name: str) -> Dict:
    """
    Generate a single training example from multiple reviews
    
    Args:
        reviews: List of review dictionaries with 'review_body' and 'rating'
        restaurant_name: Name of the restaurant
        
    Returns:
        Dictionary with 'input' and 'output' keys
    """
    # Combine all reviews into one text
    combined_reviews = " ".join([r['review_body'] for r in reviews])
    
    # Create input text
    input_text = f"analyze_food: Reviews from {restaurant_name}: {combined_reviews}"
    
    # Extract food mentions and their sentiments
    food_sentiments = defaultdict(lambda: {"positive": 0, "negative": 0, "total": 0})
    
    for review in reviews:
        review_text = review['review_body']
        rating = review.get('rating', None)
        sentiment = determine_sentiment(review_text, rating)
        
        # Extract food mentions from this review
        foods = extract_food_mentions(review_text)
        
        for food in foods:
            food_sentiments[food]["total"] += 1
            if sentiment == "positive":
                food_sentiments[food]["positive"] += 1
            elif sentiment == "negative":
                food_sentiments[food]["negative"] += 1
    
    # Sort foods by total mentions
    sorted_foods = sorted(food_sentiments.items(), key=lambda x: x[1]["total"], reverse=True)
    
    # Get top 3 best (most positive) and worst (most negative)
    best_foods = []
    worst_foods = []
    
    for food, counts in sorted_foods:
        if counts["total"] > 0:
            pos_pct = (counts["positive"] / counts["total"]) * 100
            neg_pct = (counts["negative"] / counts["total"]) * 100
            
            if pos_pct >= 60:  # At least 60% positive
                best_foods.append((food, counts["total"], pos_pct))
            elif neg_pct >= 60:  # At least 60% negative
                worst_foods.append((food, counts["total"], neg_pct))
    
    # Limit to top 3 each
    best_foods = best_foods[:3]
    worst_foods = worst_foods[:3]
    
    # Generate output text
    output_parts = []
    
    if best_foods:
        best_str = ", ".join([f"{food} ({count} mentions, {pct:.0f}% positive)" 
                             for food, count, pct in best_foods])
        output_parts.append(f"Top 3 best: {best_str}")
    
    if worst_foods:
        worst_str = ", ".join([f"{food} ({count} mentions, {pct:.0f}% negative)" 
                              for food, count, pct in worst_foods])
        output_parts.append(f"Top 3 worst: {worst_str}")
    
    output_text = ". ".join(output_parts) + "."
    
    # If no foods found, generate a default response
    if not output_parts:
        output_text = "No specific food items mentioned in reviews."
    
    return {
        "input": input_text,
        "output": output_text,
        "restaurant_name": restaurant_name,
        "num_reviews": len(reviews)
    }


def generate_training_data(reviews_df: pd.DataFrame, 
                          reviews_per_example: int = 5,
                          max_examples: int = 1000) -> List[Dict]:
    """
    Generate training data from scraped reviews
    
    Args:
        reviews_df: DataFrame with reviews
        reviews_per_example: Number of reviews to combine per training example
        max_examples: Maximum number of training examples to generate
        
    Returns:
        List of training examples
    """
    print("\n" + "="*60)
    print("GENERATING TRAINING DATA")
    print("="*60)
    
    training_data = []
    
    # Group reviews by restaurant
    for restaurant_name, group in reviews_df.groupby('restaurant_name'):
        print(f"\nProcessing {restaurant_name}...")
        
        # Convert to list of dicts
        reviews = group.to_dict('records')
        
        # Generate training examples by combining reviews
        num_examples = min(len(reviews) // reviews_per_example, max_examples // reviews_df['restaurant_name'].nunique())
        
        for i in range(num_examples):
            start_idx = i * reviews_per_example
            end_idx = min(start_idx + reviews_per_example, len(reviews))
            
            if end_idx - start_idx >= 2:  # Need at least 2 reviews
                example_reviews = reviews[start_idx:end_idx]
                example = generate_training_example(example_reviews, restaurant_name)
                training_data.append(example)
    
    print(f"\nGenerated {len(training_data)} training examples")
    
    return training_data


def save_training_data(training_data: List[Dict], output_path: str = "data/training_data.json"):
    """
    Save training data to JSON file
    
    Args:
        training_data: List of training examples
        output_path: Path to save the JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved training data to {output_path}")
    print(f"Total examples: {len(training_data)}")


def load_training_data(input_path: str = "data/training_data.json") -> List[Dict]:
    """
    Load training data from JSON file
    
    Args:
        input_path: Path to the JSON file
        
    Returns:
        List of training examples
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    print(f"Loaded {len(training_data)} training examples from {input_path}")
    return training_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate training data for mT5 model")
    parser.add_argument("--input", default="data/raw_reviews/all_restaurants.csv", 
                       help="Input CSV file with reviews")
    parser.add_argument("--output", default="data/training_data.json",
                       help="Output JSON file for training data")
    parser.add_argument("--reviews-per-example", type=int, default=5,
                       help="Number of reviews to combine per training example")
    parser.add_argument("--max-examples", type=int, default=1000,
                       help="Maximum number of training examples")
    
    args = parser.parse_args()
    
    # Load reviews
    try:
        reviews_df = pd.read_csv(args.input)
        print(f"Loaded {len(reviews_df)} reviews from {args.input}")
    except FileNotFoundError:
        print(f"Error: File {args.input} not found!")
        print("Please run scraper.py first with --sample flag to create sample data.")
        exit(1)
    
    # Generate training data
    training_data = generate_training_data(
        reviews_df, 
        reviews_per_example=args.reviews_per_example,
        max_examples=args.max_examples
    )
    
    # Save training data
    save_training_data(training_data, args.output)
    
    # Show some examples
    print("\n" + "="*60)
    print("SAMPLE TRAINING EXAMPLES")
    print("="*60)
    
    for i, example in enumerate(training_data[:3]):
        print(f"\nExample {i+1}:")
        print(f"Input: {example['input'][:200]}...")
        print(f"Output: {example['output']}")
