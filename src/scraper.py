"""
Google Maps Review Scraper for Buffalo Restaurants
Scrapes reviews from Google Maps for fine-tuning the mT5 model
"""

import os
import pandas as pd
from datetime import datetime
from typing import List, Dict


# Target restaurants in Buffalo, NY
RESTAURANTS = {
    "Bar-Bill Tavern": "https://www.google.com/maps/place/Bar-Bill+Tavern/@42.8214586,-78.8329963,17z/data=!3m1!4b1!4m6!3m5!1s0x89d45427a3670d11:0x3ea73ea2e2e2e2e2!8m2!3d42.8214586!4d-78.8308076",
    "Gabriel's Gate": "https://www.google.com/maps/place/Gabriel's+Gate/@42.8864586,-78.8789963,17z/data=!3m1!4b1!4m6!3m5!1s0x89d45427a3670d11:0x3ea73ea2e2e2e2e3!8m2!3d42.8864586!4d-78.8768076",
    "Ristorante Lombardo": "https://www.google.com/maps/place/Ristorante+Lombardo/@42.9264586,-78.8189963,17z/data=!3m1!4b1!4m6!3m5!1s0x89d45427a3670d11:0x3ea73ea2e2e2e2e4!8m2!3d42.9264586!4d-78.8168076",
    "Picasso's Pizza": "https://www.google.com/maps/place/Picasso's+Pizza/@42.9064586,-78.8589963,17z/data=!3m1!4b1!4m6!3m5!1s0x89d45427a3670d11:0x3ea73ea2e2e2e2e5!8m2!3d42.9064586!4d-78.8568076",
    "Oliver's Restaurant": "https://www.google.com/maps/place/Oliver's+Restaurant/@42.8964586,-78.8489963,17z/data=!3m1!4b1!4m6!3m5!1s0x89d45427a3670d11:0x3ea73ea2e2e2e2e6!8m2!3d42.8964586!4d-78.8468076"
}


def scrape_restaurant_reviews(restaurant_name: str, url: str, limit: int = 500) -> pd.DataFrame:
    """
    Scrape reviews for a single restaurant from Google Maps
    
    Args:
        restaurant_name: Name of the restaurant
        url: Google Maps URL for the restaurant
        limit: Maximum number of reviews to scrape
        
    Returns:
        DataFrame with reviews
    """
    print(f"\n{'='*60}")
    print(f"Scraping reviews for: {restaurant_name}")
    print(f"{'='*60}")
    
    try:
        # Use the google-maps-reviews-scraper CLI
        import subprocess
        
        output_file = f"data/raw_reviews/{restaurant_name.lower().replace(' ', '_')}.csv"
        
        # Run the scraper
        cmd = [
            "gmaps-reviews", "scrape", url,
            "--csv", output_file,
            "--limit", str(limit)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error scraping {restaurant_name}: {result.stderr}")
            return pd.DataFrame()
        
        # Load the scraped data
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            df['restaurant_name'] = restaurant_name
            print(f"Successfully scraped {len(df)} reviews for {restaurant_name}")
            return df
        else:
            print(f"No output file found for {restaurant_name}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error scraping {restaurant_name}: {str(e)}")
        return pd.DataFrame()


def scrape_all_restaurants(limit_per_restaurant: int = 500) -> pd.DataFrame:
    """
    Scrape reviews for all target restaurants
    
    Args:
        limit_per_restaurant: Maximum reviews per restaurant
        
    Returns:
        Combined DataFrame with all reviews
    """
    print("\n" + "="*60)
    print("BUFFALO RESTAURANT REVIEW SCRAPER")
    print("="*60)
    
    all_reviews = []
    
    for restaurant_name, url in RESTAURANTS.items():
        df = scrape_restaurant_reviews(restaurant_name, url, limit_per_restaurant)
        if not df.empty:
            all_reviews.append(df)
    
    if all_reviews:
        combined_df = pd.concat(all_reviews, ignore_index=True)
        
        # Save combined dataset
        output_path = "data/raw_reviews/all_restaurants.csv"
        combined_df.to_csv(output_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"SCRAPING COMPLETE")
        print(f"{'='*60}")
        print(f"Total reviews scraped: {len(combined_df)}")
        print(f"Restaurants: {combined_df['restaurant_name'].nunique()}")
        print(f"Saved to: {output_path}")
        
        return combined_df
    else:
        print("No reviews were scraped!")
        return pd.DataFrame()


def create_sample_data():
    """
    Create sample data for testing when scraping is not available
    This allows testing the pipeline without actual scraping
    """
    print("\nCreating sample data for testing...")
    
    sample_reviews = [
        # Bar-Bill Tavern reviews
        {"restaurant_name": "Bar-Bill Tavern", "review_body": "The wings here are incredible! Best I've ever had in Buffalo. The honey butter barbecue flavor is amazing.", "rating": 5, "review_title": "Amazing wings!"},
        {"restaurant_name": "Bar-Bill Tavern", "review_body": "Beef on weck was perfectly done. The roast beef was tender and the roll was fresh. Highly recommend!", "rating": 5, "review_title": "Perfect beef on weck"},
        {"restaurant_name": "Bar-Bill Tavern", "review_body": "Wings were okay but nothing special. The fries were soggy and cold. Disappointing visit.", "rating": 2, "review_title": "Disappointing visit"},
        {"restaurant_name": "Bar-Bill Tavern", "review_body": "Great spot! Wings are crispy and the sauce is perfect. Will definitely come back.", "rating": 5, "review_title": "Great spot!"},
        {"restaurant_name": "Bar-Bill Tavern", "review_body": "The salad was fresh but overpriced for what you get. Wings are the real star here.", "rating": 4, "review_title": "Wings are the star"},
        
        # Gabriel's Gate reviews
        {"restaurant_name": "Gabriel's Gate", "review_body": "Best wings in Allentown! Crispy outside, juicy inside. The Frank's Red Hot is perfect.", "rating": 5, "review_title": "Best wings in town!"},
        {"restaurant_name": "Gabriel's Gate", "review_body": "Love the atmosphere here. Wings are consistently good. The beef on weck is also excellent.", "rating": 5, "review_title": "Consistently good"},
        {"restaurant_name": "Gabriel's Gate", "review_body": "Wings were decent but took forever to come out. Service was slow.", "rating": 3, "review_title": "Slow service"},
        {"restaurant_name": "Gabriel's Gate", "review_body": "The wings are overrated. I've had better elsewhere. Fries were cold.", "rating": 2, "review_title": "Overrated wings"},
        {"restaurant_name": "Gabriel's Gate", "review_body": "Great place for wings! The sauce selection is amazing. Try the medium heat.", "rating": 5, "review_title": "Great sauce selection"},
        
        # Ristorante Lombardo reviews
        {"restaurant_name": "Ristorante Lombardo", "review_body": "Best Italian in Buffalo! The pasta is homemade and delicious. The veal parmesan was perfect.", "rating": 5, "review_title": "Best Italian in Buffalo!"},
        {"restaurant_name": "Ristorante Lombardo", "review_body": "The octopus appetizer was amazing. Main courses were excellent. Great wine list.", "rating": 5, "review_title": "Excellent Italian"},
        {"restaurant_name": "Ristorante Lombardo", "review_body": "Food was good but service was slow. The pasta al dente was perfect though.", "rating": 4, "review_title": "Good food, slow service"},
        {"restaurant_name": "Ristorante Lombardo", "review_body": "Overpriced for what you get. The salad was disappointing. Pasta was the only good thing.", "rating": 2, "review_title": "Overpriced"},
        {"restaurant_name": "Ristorante Lombardo", "review_body": "Amazing dining experience! The gnocchi was light and fluffy. The tiramisu was the best I've ever had.", "rating": 5, "review_title": "Amazing experience"},
        
        # Picasso's Pizza reviews
        {"restaurant_name": "Picasso's Pizza", "review_body": "Best pizza in Buffalo! The cup-and-char pepperoni is perfect. Crispy crust, great sauce.", "rating": 5, "review_title": "Best pizza in Buffalo!"},
        {"restaurant_name": "Picasso's Pizza", "review_body": "Love their pizza! The cheese is perfectly melted and the pepperoni cups are amazing.", "rating": 5, "review_title": "Perfect pizza"},
        {"restaurant_name": "Picasso's Pizza", "review_body": "Pizza was okay but the wings were terrible. Dry and overcooked.", "rating": 3, "review_title": "Pizza good, wings bad"},
        {"restaurant_name": "Picasso's Pizza", "review_body": "The pizza is overrated. Crust was soggy and the sauce was bland.", "rating": 2, "review_title": "Overrated pizza"},
        {"restaurant_name": "Picasso's Pizza", "review_body": "Great pizza! The crust is perfectly crispy. The pepperoni is the best I've had.", "rating": 5, "review_title": "Great crust!"},
        
        # Oliver's Restaurant reviews
        {"restaurant_name": "Oliver's Restaurant", "review_body": "Fine dining at its best! The steak was cooked to perfection. The wine pairings were excellent.", "rating": 5, "review_title": "Fine dining perfection"},
        {"restaurant_name": "Oliver's Restaurant", "review_body": "Amazing tasting menu! Every course was perfectly executed. The dessert was heavenly.", "rating": 5, "review_title": "Amazing tasting menu"},
        {"restaurant_name": "Oliver's Restaurant", "review_body": "Food was excellent but the service was pretentious. The fish dish was perfect though.", "rating": 4, "review_title": "Excellent food"},
        {"restaurant_name": "Oliver's Restaurant", "review_body": "Overpriced and underwhelming. The salad was wilted and the main course was small.", "rating": 2, "review_title": "Overpriced and underwhelming"},
        {"restaurant_name": "Oliver's Restaurant", "review_body": "Special occasion perfect! The beef wellington was incredible. The chocolate dessert was to die for.", "rating": 5, "review_title": "Special occasion perfect"},
    ]
    
    df = pd.DataFrame(sample_reviews)
    
    # Save sample data
    output_path = "data/raw_reviews/all_restaurants.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Created {len(df)} sample reviews")
    print(f"Saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape Google Maps reviews for Buffalo restaurants")
    parser.add_argument("--sample", action="store_true", help="Use sample data instead of scraping")
    parser.add_argument("--limit", type=int, default=500, help="Max reviews per restaurant")
    
    args = parser.parse_args()
    
    if args.sample:
        create_sample_data()
    else:
        scrape_all_restaurants(limit_per_restaurant=args.limit)
