from django.shortcuts import render
import urllib.parse
import pymysql
from sqlalchemy import create_engine
import pandas as pd
import math
import json
from django.template.loader import render_to_string
# NLTK
import nltk
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create your views here.

# Download the NLTK resources for stopwords and the Porter Stemmer
nltk.download('stopwords')
nltk.download('punkt')

def dbconnect():
    username='root'
    password=urllib.parse.quote_plus('Pass@123')
    host='127.0.0.1'
    database='coventry'
    table_name='CGL'
    engine = create_engine('mysql+pymysql://'+username+':'+password+'@'+host+'/'+database)
    try:
        engine.connect()
        print("Connected Successfully")
        return engine
    except exc.SQLAlchemyError as err:
        print("error", err.__cause__)
def search(request):
    context={}
    engine=dbconnect()
    if request.method == 'POST':
        search = request.POST['q']
        # Read Data from MySQL
        df = pd.read_sql_query("SELECT * FROM CGL",con=engine)
        # Pre-Processing Data
        # Stop Word and Stemmer
        stop_words = set(stopwords.words("english"))
        stemmer = PorterStemmer()
        # Preprocess the text data for each paper
        df["combined_text"] = df['Title'].str.lower() + " " + df[ 'Published_year'] + " " + df['Author'].str.lower()
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words="english", use_idf=True)       
        # Calculate the TF-IDF scores for each paper
        tfidf_matrix = vectorizer.fit_transform(df["combined_text"])
        # Get the feature names (stems) from the vectorizer
        # stems = vectorizer.get_feature_names_out()
        stems = vectorizer.get_feature_names()
        # Create a DataFrame with the TF-IDF scores for each paper
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=stems)        
        # Concatenate the original DataFrame with the TF-IDF DataFrame
        ranking_df = pd.concat([df, tfidf_df], axis=1)

        # Query Processing
        query = search
        # Tokenize and preprocess the search query
        query_tokens = [stemmer.stem(token) for token in word_tokenize(query.lower()) if token not in stop_words]
        # Compute the scores for each paper based on the search query
        scores = defaultdict(float)
        for i, row in ranking_df.iterrows():
            all_tokens = [stemmer.stem(token) for token in word_tokenize(row["combined_text"]) if token not in stop_words]
            for token in query_tokens:
                if token in all_tokens:
                    scores[i] += 1

        # Sort the papers by score
        top_papers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Get the paper information
        results = []
        for paper_id, score in top_papers:
            paper = ranking_df.iloc[paper_id]
            results.append({
                    "title": paper['Title'],
                    "title_link": paper['Title_Link'],
                    "year": paper['Published_year'],
                    "authors": paper['Author'],
                    "author_links": paper['Author_link']
                })
        resulted_df = pd.DataFrame.from_dict(results)
        print(resulted_df.columns)
        json_records = resulted_df.reset_index().to_json(orient='records')
        data=[]
        data = json.loads(json_records)
        context['table_1'] = render_to_string('render_table.html',context={'table_1':data})
    return render(request,'search.html',context)