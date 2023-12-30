import streamlit as st
import pickle
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
from recommend import recommend
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize,pos_tag
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import pytesseract
from PIL import Image,ImageEnhance, ImageFilter

#setting streamlit page
with st.sidebar:
    st.title("FINAL PROJECT")
    selected= option_menu("Main Menu",["Customer convertion prediction","Movie recommandation system","NLP","Image processing"])
if selected == "Customer convertion prediction":
    # Load the trained model
    with open("log_reg_model.pkl", "rb") as mf:
        model = pickle.load(mf)
    transactionRevenue = st.text_input("Transaction Revenue", 0)
    time_on_site = st.text_input("Time on Site", 7980)
    historic_session_page = st.text_input("Historic Session Page", 0)
    products_array = st.text_input("Products Array", 5874)
    avg_session_time = st.text_input("Average Session Time", 742.795489)
    if st.button("Predict"):
        # Convert input values to float
        input_data = np.array([
            float(transactionRevenue),
            float(time_on_site),
            float(historic_session_page),
            float(products_array),
            float(avg_session_time)
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Display prediction
        if prediction != 0:
            st.success("Converted as a customer!")
        else:
            st.success("Not converted as a customer.")
if selected == "Movie recommandation system":
    movie_input = st.text_input("Enter a movie name for recommendation", "Jumanji")
    if st.button("Recommend"):
        recommended_movies = recommend(movie_input)
        st.table(recommended_movies)
if selected == "NLP":
    senti = st.text_input("Enter any text for Sentiment Analysis")
    if st.button("Process"):
        process = TextBlob(senti)
        value = process.sentiment
        if value.polarity > 0.5:
            st.success("Positive")
        else:
            st.success("Negative")
    stems_input = st.text_input("Enter text for stemming")
    if st.button("Execute Stemming"):
        stemmer = PorterStemmer()
        words = word_tokenize(stems_input)
        stemmed_words = [stemmer.stem(word) for word in words]
        st.write("Stemmed Text:")
        st.write(" ".join(stemmed_words))
    lem_input = st.text_input("Enter text for Lemmatization")
    if st.button("Execute lemmatization"):
        def lemmatization(text):
            result=[]
            wordnet = WordNetLemmatizer()
            word = word_tokenize(lem_input)
            for token,tag in pos_tag(word):
                pos=tag[0].lower()
                if pos not in ['a', 'r', 'n', 'v']:
                    pos='n'
                result.append(wordnet.lemmatize(token,pos))
            return result
        lemmatized_words = lemmatization(lem_input)
        st.write("Lemmatized Text:")
        st.write(" ".join(lemmatized_words))
if selected == "Image processing":
    #text extraction of image
    file= st.file_uploader("Upload Image for text extraction",type=["png","jpg","jpeg"])
    if file is not None:
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract' 
        extractedInformation = pytesseract.image_to_string(Image.open(file))
        st.image(Image.open(file),caption='Text extraction',use_column_width=True)
        st.success(extractedInformation)
    else:
        st.warning("Please upload an image.")


    uploader = st.file_uploader("upload image for preprocessing",type=["png","jpg","jpeg"])
    if uploader is not None:
        upload_image=Image.open(uploader)
        #brightening the image
        brightness_factor = st.slider("Brightness", 0.0, 6.0, 1.0)
        enhanced_image = ImageEnhance.Brightness(upload_image).enhance(brightness_factor)
        #bluring image
        blur_radius = st.slider("Blur", 0, 10, 0)
        blurred_image = upload_image.filter(ImageFilter.BoxBlur(blur_radius))
        #edge detection
        edge_detected_image = upload_image.filter(ImageFilter.FIND_EDGES)
        #display images
        st.image([enhanced_image, blurred_image, edge_detected_image], caption=["Enhanced", "Blurred", "Edge Detection"], use_column_width=True)
    else:
        st.warning("Please upload an image for preprocessing.")

