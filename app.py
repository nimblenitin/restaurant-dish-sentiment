"""
Streamlit UI for Restaurant Food Recommendation System
Interactive web interface for testing the fine-tuned mT5 model
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from inference import FoodRecommender, format_output


# Page configuration
st.set_page_config(
    page_title="Restaurant Food Recommendation",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ff6b6b;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .recommendation-box {
        background-color: #f0f8ff;
        border: 2px solid #4a90d9;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .best-food {
        color: #28a745;
        font-weight: bold;
    }
    .worst-food {
        color: #dc3545;
        font-weight: bold;
    }
    .stButton > button {
        background-color: #4a90d9;
        color: white;
        font-weight: bold;
        padding: 0.5rem 2rem;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #357abd;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the fine-tuned model (cached for performance)"""
    try:
        # Try to load fine-tuned model first
        model_path = "models/mt5-finetuned-restaurant-food"
        if os.path.exists(model_path) and os.listdir(model_path):
            recommender = FoodRecommender(model_path)
        else:
            # Fall back to base model
            st.info("Using base mT5 model. Train a custom model for better results!")
            recommender = FoodRecommender()
        return recommender
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please install all dependencies: pip install -r requirements.txt")
        return None


def main():
    # Header
    st.markdown('<h1 class="main-header">🍽️ Restaurant Food Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Fine-tuned mT5 model for analyzing restaurant reviews</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        This system analyzes restaurant reviews and recommends:
        - **Top 3 Best Dishes** (most praised)
        - **Top 3 Worst Dishes** (most criticized)
        
        With mention counts and sentiment percentages.
        """)
        
        st.header("Target Restaurants")
        restaurants = [
            "Bar-Bill Tavern",
            "Gabriel's Gate",
            "Ristorante Lombardo",
            "Picasso's Pizza",
            "Oliver's Restaurant"
        ]
        for restaurant in restaurants:
            st.markdown(f"- {restaurant}")
        
        st.header("Instructions")
        st.markdown("""
        1. Select or enter restaurant name
        2. Add reviews (one per line)
        3. Click **Get Recommendations**
        4. View the food recommendations!
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Reviews")
        
        # Restaurant selection
        restaurant_name = st.text_input(
            "Restaurant Name",
            value="Bar-Bill Tavern",
            help="Enter the name of the restaurant"
        )
        
        # Review input
        reviews_text = st.text_area(
            "Enter Reviews (one per line)",
            height=300,
            placeholder="The wings here are incredible! Best I've ever had.\nThe beef on weck was perfectly done.\nThe salad was disappointing and overpriced.",
            help="Enter multiple reviews, one per line"
        )
        
        # Number of reviews info
        if reviews_text:
            review_count = len([r for r in reviews_text.strip().split('\n') if r.strip()])
            st.info(f"📝 {review_count} review(s) entered")
        
        # Action buttons
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            analyze_button = st.button("🔍 Get Recommendations", type="primary", use_container_width=True)
        
        with col_btn2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    with col2:
        st.header("🎯 Recommendations")
        
        if clear_button:
            st.rerun()
        
        if analyze_button:
            if not reviews_text.strip():
                st.warning("⚠️ Please enter at least one review!")
            else:
                # Parse reviews
                reviews = [r.strip() for r in reviews_text.strip().split('\n') if r.strip()]
                
                if not reviews:
                    st.warning("⚠️ Please enter valid reviews!")
                else:
                    # Load model
                    recommender = load_model()
                    
                    if recommender:
                        with st.spinner("🔄 Analyzing reviews..."):
                            try:
                                # Get recommendation
                                recommendation = recommender.analyze_reviews(
                                    reviews, 
                                    restaurant_name
                                )
                                
                                # Display result
                                st.success("✅ Analysis complete!")
                                
                                # Styled recommendation box
                                st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                                st.markdown("### 🍴 Food Recommendations")
                                st.markdown(format_output(recommendation))
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Additional info
                                st.markdown("---")
                                st.markdown("### 📊 Analysis Details")
                                
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Restaurant", restaurant_name)
                                with col_b:
                                    st.metric("Reviews Analyzed", len(reviews))
                                
                            except Exception as e:
                                st.error(f"Error during analysis: {e}")
        
        # Demo section
        if not analyze_button:
            st.markdown("---")
            st.markdown("### 🎮 Try Demo")
            
            if st.button("📦 Load Sample Reviews", use_container_width=True):
                sample_reviews = [
                    "The wings here are incredible! Best I've ever had in Buffalo. The honey butter barbecue flavor is amazing.",
                    "Beef on weck was perfectly done. The roast beef was tender and the roll was fresh. Highly recommend!",
                    "The salad was fresh but overpriced for what you get. Wings are the real star here.",
                    "Wings were okay but nothing special. The fries were soggy and cold. Disappointing visit.",
                    "Great spot! Wings are crispy and the sauce is perfect. Will definitely come back."
                ]
                st.session_state['sample_reviews'] = "\n".join(sample_reviews)
                st.session_state['sample_restaurant'] = "Bar-Bill Tavern"
                st.rerun()
            
            # Load sample if available
            if 'sample_reviews' in st.session_state:
                st.text_area(
                    "Sample Reviews",
                    value=st.session_state['sample_reviews'],
                    height=200,
                    disabled=True,
                    key="sample_display"
                )
                st.info("👆 Click **Get Recommendations** to analyze these reviews!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        Built with ❤️ using Streamlit and Hugging Face Transformers<br>
        Fine-tuned mT5 model for restaurant food recommendations
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
