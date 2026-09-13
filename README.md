
<h1>Restaurant Dish Recommender</h1>

## Overview
This project helps users find the best dishes at their favorite restaurants by analyzing Google Maps reviews using advanced Natural Language Processing (NLP) techniques. By inputting a Google Maps restaurant link, the app identifies the top dishes in terms of user rating, popularity, and health. Restaurant Dish Recommender eliminates the need for users to spend time sifting through countless reviews to find the best dishes, making it easier for them to make informed decisions about what to order. Additionally, the health score shows the calorie content of each dish, promoting a balanced diet and overall healthier lifestyle.

## How it Works
Restaurant Dish Recommender scrapes reviews from Google Maps and analyzes them using advanced NLP techniques. Specifically, named entity recognition (NER) is used to identify all food entities mentioned in the reviews. The data is then processed to ensure that only actual menu items are considered, eliminating ingredients like soup, broth, or spices that are often mentioned in reviews.

Once a list of actual menu items is obtained, each food's sentiment is analyzed using aspect-based sentiment analysis (ABSA). This approach determines how much the reviewer enjoyed each dish and attributes a score to it accordingly. Based on this analysis, points are added or subtracted in the scoring algorithm for each dish, which helps determine the top 3 dishes in terms of user rating.

In addition to user ratings, Restaurant Dish Recommender also analyzes food popularity among users. A similar process is used, but instead of sentiment analysis, the frequency of each food item mentioned in reviews is the focus. This determines the top 3 dishes in terms of popularity among users.

Healthy eating is also taken into account. A health score for each dish is incorporated using data from a national government database. This allows users to view the health scores for the top 3 dishes, making it easier for them to choose the healthiest options available at their favorite restaurants and maintain a balanced diet.

## Usage
<p> To run this project on your own you must do the following steps 
<br></br>
  <b>1.</b> Navigate to a command line and clone the repository 
</p>

```
git clone https://github.com/nimblenitin/restaurant-dish-sentiment.git
```
<p>
  <b>2.</b> Install dependencies and start the backend
</p>

```
cd restaurant-dish-sentiment
pip3 install flask flask-cors requests transformers huggingface_hub
export foodkey="your_usda_api_key"
python main.py
```
<p>
  <b>3.</b> In a new terminal, start the frontend
</p>

```
cd restaurant-dish-sentiment/front-end
npm install
npm run dev
```
<p>
  Open http://localhost:5173 in your browser and paste a Google Maps restaurant URL.
</p>

## Technologies used
  Front-end: <b>[React.js](https://github.com/facebook/react/blob/main/LICENSE)</b> \
  React is used for the entire webview of the project \
  \
  Back-end: <b>[Flask](https://github.com/pallets/flask/blob/main/LICENSE.rst "Flask license")</b> \
  Flask is used to connect the data processing to the front-end \
  \
  Web Scraping: <b>[Playwright](https://playwright.dev/)</b> \
  Playwright is used to scrape Google Maps reviews via a headless browser \
  \
  Named-entity recognition: <b>[InstaFoodRoBERTa-NER](https://huggingface.co/Dizex/InstaFoodRoBERTa-NER "InstaFoodROBERTa-NER")</b> \
  Dizex's fine-tuned BERT model is used to recognize food entities in reviews \
  \
  Aspect-based sentiment analysis: <b>[deberta-v3-large-absa-v1.1](https://huggingface.co/yangheng/deberta-v3-large-absa-v1.1 "deberta-v3-large-absa-v1.1")</b> \
  Yangheng's ABSA model is used to determine the sentiment of a specific food in a review \
  \
  Food database: <b>[FoodData Central](https://fdc.nal.usda.gov/ "FoodData Central")</b> \
  The U.S. Department of Agriculture's food database is used to find the calories of foods and determine the health scores for them
