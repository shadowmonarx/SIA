from bs4 import BeautifulSoup
from matplotlib import colors
import requests
import csv
import time
import os
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from tqdm import tqdm
import urllib.parse
import matplotlib.pyplot as plt
from IPython.display import Image, display
import streamlit as st
from PIL import Image
from io import BytesIO

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36",
    "Accept-Language": "en-US, en;q=0.5"
}

def get_first_product_details(search_query):
    url = f"https://www.flipkart.com/search?q={urllib.parse.quote_plus(search_query)}"
    response = requests.get(url, headers=HEADERS)
    
    if response.ok:
        page_soup = BeautifulSoup(response.text, 'html.parser')
        first_product = page_soup.find('a', class_='_1fQZEK')
        if first_product and 'href' in first_product.attrs:
            product_url = 'https://www.flipkart.com' + first_product.attrs['href']
            product_name = first_product.find('div', class_='_4rR01T').text
            product_price = first_product.find('div', class_='_30jeq3 _1_WHN1').text
            image_element = first_product.find('img', {'class': '_396cs4'})
            if image_element:
                product_image_url = image_element['src']
            else:
                product_image_url = None            
            return product_url, product_name, product_price, product_image_url
        return None, None, None, None

def get_all_reviews_link(product_url):
    response = requests.get(product_url, headers=HEADERS)
    if response.ok:
        page_soup = BeautifulSoup(response.text, 'html.parser')
        all_reviews_button = page_soup.find('div', class_='_3UAT2v _16PBlm')
        if all_reviews_button and all_reviews_button.parent.name == 'a':
            return 'https://www.flipkart.com' + all_reviews_button.parent['href']
    return None

def scrape_all_reviews(all_reviews_url):
    reviews = []
    while True and len(reviews) < 200: 
        response = requests.get(all_reviews_url, headers=HEADERS)
        if not response.ok:
            break
        
        page_soup = BeautifulSoup(response.text, 'html.parser')
        review_containers = page_soup.find_all('div', class_='t-ZTKy')
        for container in review_containers:
            review_text = container.get_text(strip=True).replace("READ MORE", "")
            reviews.append(review_text)
        
        next_page = page_soup.find('a', {'class': '_1LKTO3'}, string=lambda x: x and 'Next' in x)
        if next_page and 'href' in next_page.attrs:
            all_reviews_url = 'https://www.flipkart.com' + next_page.attrs['href']
        else:
            break
        
    return reviews


def analyze_and_save_sentiments(reviews, product_name):
    sia = SentimentIntensityAnalyzer()
    sentiment_data = []

    for review in tqdm(reviews, desc="Analyzing Sentiment"):
        scores = sia.polarity_scores(review)
        sentiment = 'Positive' if scores['compound'] >= 0.05 else 'Negative' if scores['compound'] <= -0.05 else 'Neutral'
        sentiment_data.append({
            'Review': review,
            'Neg': scores['neg'],
            'Neu': scores['neu'],
            'Pos': scores['pos'],
            'Compound': scores['compound'],
            'Sentiment': sentiment
        })

    df_sentiment = pd.DataFrame(sentiment_data)
    sentiment_csv_filename = f"{product_name.replace(' ', '_').replace('/', '_')}_sentiment.csv"
    df_sentiment.to_csv(sentiment_csv_filename, index=False)
    print(f"Sentiment analysis saved to {sentiment_csv_filename}")

def plot_sentiment_distribution(sentiment_csv_filenames):

    
    for filename in sentiment_csv_filenames:
        df = pd.read_csv(filename)
        sentiment_counts = df['Sentiment'].value_counts(normalize=True) * 100

        fig, ax = plt.subplots(figsize=(6, 6))
        wedges, texts, autotexts = ax.pie(sentiment_counts, 
                                           labels=sentiment_counts.index,
                                           autopct='%1.1f%%', 
                                           startangle=140, 
                                           )

    
        ax.set_facecolor('none')
        ax.axis('equal')

        plt.title(f'Sentiment Analysis for {filename.split("_sentiment")[0].replace("_", " ")}', fontdict={'fontsize':18, 'fontweight':'bold'})

        plt.box(False)

        st.pyplot(fig)

def plot_price_comparison(product_details):
    products = [detail['name'] for detail in product_details]
    prices = [float(detail['price'].replace('₹', '').replace(',', '')) if isinstance(detail['price'], str) else detail['price'] for detail in product_details]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(products, prices, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.xlabel('Products')
    plt.ylabel('Price in INR')
    plt.title('Price Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)

def compare_products(product_details):
    # Convert price string to float and calculate scores
    for product in product_details:
        if isinstance(product['price'], str):
            product['price'] = float(product['price'].replace('₹', '').replace(',', ''))
        product['sentiment_score'] = product['positive_sentiment'] - product['negative_sentiment']
        product['adjusted_score'] = product['sentiment_score'] - product['price'] / 1000
        product['adjusted_score'] = round(product['adjusted_score'], 2)
    
    # Sort the products by adjusted score
    product_details.sort(key=lambda x: x['adjusted_score'], reverse=True)
    
    # Display products with images using columns
    for product in product_details:
        cols = st.columns([1, 2, 1, 2])
        with cols[0]:
            if product['image_url']:
                response = requests.get(product['image_url'])
                image = Image.open(BytesIO(response.content))
                st.image(image, width=150)  # Set width as required
            else:
                st.write("No Image")
        
        with cols[1]:
            st.text(f"Product: {product['name']}")
        
        with cols[2]:
            st.text(f"Price: ₹{product['price']:.2f}")
        
        with cols[3]:
            st.text(f"Review Score: {product['adjusted_score']:.2f}")
        
        st.markdown("---")  # Adds a horizontal line for separation
        
    # Recommend the best product
    best_product = product_details[0]
    reason = f"with the highest review score of {best_product['adjusted_score']} after accounting for its price of ₹{best_product['price']:.2f} INR."

# Create two columns
    col1, col2 = st.columns([5, 2])  # Adjust the ratio as needed for your layout

# Use the first column for the success message
    with col1:
        st.success(f"Recommendation: Buy '{best_product['name']}' {reason}")

# Use the second column for the image
    with col2:
        if best_product['image_url']:
            response = requests.get(best_product['image_url'])
            best_image = Image.open(BytesIO(response.content))
            st.image(best_image, width=150, caption=f"Recommended: {best_product['name']}")

def streamlit_app():
    st.set_page_config(
  page_title='Product Comparison Tool',
  page_icon='🔍',
  layout='wide'
)
    
    with open ('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    st.title('Product Comparison Tool')

    if 'product_searches' not in st.session_state:
        st.session_state.product_searches = ['', '']

    to_delete = []  
    for i in range(len(st.session_state.product_searches)):
        col1, col2 = st.columns([4, 1])  
        with col1:
            st.session_state.product_searches[i] = st.text_input(f"Product name {i+1}",
                                                                 value=st.session_state.product_searches[i],
                                                                 key=f"product_{i}")
        with col2:
            if st.button('Delete', key=f"delete_{i}"):
                to_delete.append(i) 

    for index in reversed(to_delete):
        del st.session_state.product_searches[index]

    if st.button('Add another product'):
        st.session_state.product_searches.append('')

    if st.button('Compare Products'):
        with st.spinner('Fetching product details and reviews...'):
            product_details = []
            sentiment_csv_filenames = []

            search_queries = list(filter(None, st.session_state.product_searches))

            for search_query in search_queries:
                product_url, product_name, product_price, product_image_url = get_first_product_details(search_query)
                if product_url and product_name and product_price:
                    all_reviews_url = get_all_reviews_link(product_url)
                    if all_reviews_url:
                        reviews = scrape_all_reviews(all_reviews_url)
                        if reviews:
                            analyze_and_save_sentiments(reviews, product_name)
                            sentiment_csv_filename = f"{product_name.replace(' ', '_').replace('/', '_')}_sentiment.csv"
                            sentiment_csv_filenames.append(sentiment_csv_filename)
                            product_details.append({
                                'name': product_name,
                                'price': product_price,
                                'image_url': product_image_url
                            })
                        else:
                            st.warning(f"No reviews found for {product_name}.")
                    else:
                        st.error(f"No 'All reviews' link found for {product_name}.")
                else:
                    st.error(f"No product found for the search query: '{search_query}'.")

            for i, product in enumerate(product_details):
                sentiment_data = pd.read_csv(sentiment_csv_filenames[i])
                product['positive_sentiment'] = (sentiment_data['Sentiment'] == 'Positive').sum()
                product['negative_sentiment'] = (sentiment_data['Sentiment'] == 'Negative').sum()
                product['neutral_sentiment'] = (sentiment_data['Sentiment'] == 'Neutral').sum()

            if product_details:
                plot_sentiment_distribution(sentiment_csv_filenames)
                plot_price_comparison(product_details)
                compare_products(product_details)
                



if __name__ == '__main__':
    streamlit_app()