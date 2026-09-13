# top and worst k dish in a restaurant 

Scrapes Google reviews from restaurants and recommends the best and worst dishes based on customer sentiment.

## Design

-> `src/scraper.py` is the python file which handles scraping Google Maps reviews for Buffalo restaurants.

-> `src/label_generator.py` is the python file that processes reviews and generates training data by extracting food mentions and sentiment.

-> `src/preprocess.py` is the python file for tokenizing and preparing data for the mT5 model.

-> `src/train.py` is the python file which fine-tunes the mT5 transformer model on our restaurant review dataset.

-> `src/inference.py` is the python file that loads the trained model and generates food recommendations from new reviews.

-> `app.py` is the Streamlit web application for testing the model through a user-friendly interface.

-> `run.py` is the entry point that runs the complete pipeline from data collection to inference.

-> `test.py` is the python file to verify all components are working correctly.

