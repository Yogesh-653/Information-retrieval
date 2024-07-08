Part 1. Search Engine
1. Crawler:

1.1 Number of staff whose publications are crawled (approximately)
1. 681 papers were extracted by the crawler for the author's belongs to CGL member..
2. There were 21 staff members whose publications were crawled.
3. The author with the most publications is Katherine Wimpenny.

1.2. Information collected about each publication (e.g. links, title, year, author or any additional part)

In each publication the information that given is followed by title, author name, year that published, title links and author profile links by clicking the author name.

1.3. Which pre-processing tasks are performed before passing data to Indexer/Elastic Search 
1. Eliminate stop words from the abstract, title, and year of the tokenized work.
2.Lowercase all of the author names.
3.Tokenize the paper's author, year, and title into separate words.
4.Remove any symbols from the document title that aren't alphanumeric.

1.4 When the crawler operates, e.g. scheduled or run manually
The crawler operates manually.

1.5. Brief explanation of how it works.
For crawling and scarping code uses the request library. Web crawler that collects publication data from the Coventry University's Pure Portal website. It fetches information about authors and their publications, including titles, links, co-authors, and published years. The Crawling starts by checking the robots.txt file of the website to ensure that crawling is allowed. If crawling is not allowed (indicated by "User-Agent: *\nDisallow: /" in the robots.txt), the code exits with a message and waits for 5 seconds before checking again. If crawling is allowed, the code sets the sleep time to 0. 

The database connectivity details are provided (username, password, host, database name, and table name). It then creates a connection to the MySQL database using SQLAlchemy.The main crawling functionality begins by sending a GET request to the page containing a list of authors and their links. The page is scraped using lxml.html to extract the list of authors (CGL_authors) and their links (CGL_authors_link). The code initializes lists (CGL_author, CGL_author_link, Title, Title_links, Published_year) to store data for each author and their publications. The code loops through each author and sends a GET request to their publications page. It collects the titles (title) and links (title_link) of each publication and then loops through each publication to fetch its details, including co-authors and published year.After collecting data for all authors and publications, a DataFrame (df) is created from the collected data.The code establishes a connection with the MySQL database and saves the DataFrame (df) into a table named 'CGL' in the database. The parameter if_exists='append' ensures that the data is added to the table if it already exists.

2. Indexer

2.1. Whether you implemented the index or used Elastic Search (note that if Elastic Search is used you will lose the 15 marks for index construction, but the project becomes easier)

We haven’t undergone with Elastic Search in coding part.

2.2. If you implemented it, whether it is incremental, i.e. it grows and gets updated over the time, or it is constructed from scratch every time your crawler is run.
It is constructed from scratch every time your crawler is run.

2.3. If you implemented it, show some part of its content (e.g. the constructed dictionary)
![image](https://github.com/Yogesh-653/Information-retrieval/assets/60870157/d805626c-da3d-442b-bfae-3eec1bb3bbb7)

 
2.4. Brief explanation of how it works.

This Django view function, named search, is responsible for processing user search queries and returning search results based on the data stored in the MySQL database. The function search(request) is defined, which takes a Django request object as input and returns a render object containing the search results to be displayed on the webpage.The function first downloads the necessary NLTK resources for stopwords and the Porter Stemmer.

The dbconnect() function is defined to establish a connection with the MySQL database. It uses the SQLAlchemy library to create the engine and connect to the database. If the connection is successful, it returns the engine object.The main part of the code begins when a user submits a POST request containing the search query (q) through a form on the webpage.

The code retrieves the search query from the POST request and reads the data from the MySQL database using Pandas. It fetches the data from the CGL table and creates a DataFrame (df) from it.

Pre-processing of the text data is performed before applying the TF-IDF vectorization. Stop words are removed, and stemming is applied using the NLTK library.The text data for each paper is preprocessed and combined into a single string in the df["combined_text"] column.

The TF-IDF vectorizer is created using the TfidfVectorizer from Scikit-learn. It calculates the TF-IDF scores for each paper based on the df["combined_text"] column.

The top papers relevant to the user's search query are computed using a scoring mechanism based on the number of query tokens present in the paper's text. The papers are sorted based on the score, and the top results are stored in the results list.

The relevant information from the results list is organized into a new DataFrame called resulted_df.The resulted_df DataFrame is converted to JSON format using the to_json(orient='records') method. The resulting JSON records are stored in the data variable.The data is then rendered to an HTML template called render_table.html, which contains a table to display the search results.Finally, the render function is used to render the search.html template along with the context data, which includes the search results in the table_1 variable. This page is then sent back as a response to the user's search query


2.5 Specify whether you did the above or used Option A.
I’ve did the constraints of the indexing part.



3. Query processor:

3.1 Which pre-processing tasks are applied to a given query

Tokenization: The search query is tokenized using NLTK's word_tokenize function to break it into individual words (tokens).

Lowercasing: All the tokens in the search query are converted to lowercase using the lower() method. This ensures that the search is case-insensitive.

Stopword Removal: NLTK's stopwords corpus is used to get a set of English stopwords (common words like "the," "and," "is," etc.). These stopwords are removed from the search query to eliminate noise and improve search relevance.

Stemming: The Porter Stemmer from NLTK is used to perform stemming on the tokens. Stemming reduces words to their base or root form (e.g., "running," "runs," "ran" are all stemmed to "run"). This helps to match variations of words in the documents.

Combining Text: The Title, Published_year, and Author columns from the MySQL database are combined into a single string combined_text for each paper. This combined text will be used for further processing.

TF-IDF Vectorization: The TfidfVectorizer from Scikit-learn is used to create a TF-IDF (Term Frequency-Inverse Document Frequency) matrix for the documents. This matrix represents the importance of each term (word) in each document based on its frequency in the document and inverse frequency across the entire corpus. The TF-IDF scores are calculated for each word in the combined text of each paper.

3.2. If Elastic Search is used, how you convert a user query to an appropriate query for Elastic Search.

Elastic Search is not used in my code


3.3. If Elastic Search is NOT used, whether or not you perform ranked retrieval; if yes, specify whether or not you used vector space and the method used to calculate the ranks

The resulting TF-IDF matrix is used to compute similarity scores between the search query and the documents, and the documents are ranked based on these scores. The top-ranked papers are then returned as search results. 
TF-IDF Vectorization: The TfidfVectorizer from Scikit-learn is used to create a TF-IDF (Term Frequency-Inverse Document Frequency) matrix for the documents. This matrix represents the importance of each term (word) in each document based on its frequency in the document and inverse frequency across the entire corpus. The TF-IDF scores are calculated for each word in the combined text of each paper.

3.4. Demonstration of the running system (use screenshots in you report and run your software in your viva). You must use various input queries to prove the accuracy and robustness of your system. For example, you must use appropriate queries to prove your system performs stop-word removal and stemming and ranked retrieval.

Figure 1
Snippet of the query used in the search engine UI
![image](https://github.com/Yogesh-653/Information-retrieval/assets/60870157/ea8206af-bb1d-473a-ab9e-90eb0528fce3)



Figure 2:
Result of the query searched in search engine
![image](https://github.com/Yogesh-653/Information-retrieval/assets/60870157/37822c2d-47a1-4602-9a2d-50d0745fccac)
 

Figure 3:
Appropriate queries to prove your system performs stop-word removal:
![image](https://github.com/Yogesh-653/Information-retrieval/assets/60870157/0e4537a2-f96c-43f8-b8e0-91b9182fe794)

 
Using the same query for the stop-word removal by using the word AND between two words of the query
Figure 4:
Displaying the same result as the previous query
 ![image](https://github.com/Yogesh-653/Information-retrieval/assets/60870157/e39efc9d-c59a-4b5c-a33f-e56b5f17076b)


Figure 5:
Appropriate queries to prove your system performs the stemming
 ![image](https://github.com/Yogesh-653/Information-retrieval/assets/60870157/36955616-23b9-478e-add4-925b8c2f866e)


By using the process of stemming to the query by removing the ‘ing’ in the word of ‘Disrupting Learning’ as ‘Disrupt learn’.



Figure 6:
Result of the query as same as the previous ones.
 ![image](https://github.com/Yogesh-653/Information-retrieval/assets/60870157/d25b8785-51cc-4f88-ae7f-2a3f04560ea2)


Figure 7:
Appropriate queries to prove your system perform ranked retrieval.
 ![image](https://github.com/Yogesh-653/Information-retrieval/assets/60870157/a24a28cb-8a06-4a77-aaf8-fcc21206c5fa)


Using the query ‘disrupting learning’ for the ranking retrieval.




Figure 8:
Result of the query processing by using the TF-IDF algorithm we are come with the ranked retrieval by searching the results in the document which stands the position of words as query we searched in the search engine.
 ![image](https://github.com/Yogesh-653/Information-retrieval/assets/60870157/5a968f72-884f-4859-b983-f0c13c934b1f)

By using the query ‘disrupting learning’ the word ‘learning’ comes with ranked retrieval by each link in the result.














Figure 9:
Here is the query as author name.
![image](https://github.com/Yogesh-653/Information-retrieval/assets/60870157/11757da2-b96c-4330-8314-4e37e32ee89f)


Result 10:
Result of the query as author name with title.
 ![image](https://github.com/Yogesh-653/Information-retrieval/assets/60870157/deae5639-774d-42de-b21c-47a93345c3f6)



3.6. Brief explanation of how it works:
The template defines a search form with an input field and a search button. Users can enter their search query in the input field and click the button to initiate the search.The search results are displayed in a table with four columns: "Title," "Author," "Published Year," and "Title Link."
The CSS styles are used to format the search form and table, giving the webpage a visually appealing appearance.
The background image is set to display the background of the webpage The JavaScript code uses jQuery and the DataTables library to enhance the table display and make it interactive.
The DataTables library is initialized for the table with the ID "myTable" using the DataTable() function. This library allows sorting, filtering, and pagination for the table, making it more user-friendly and interactive. This view function processes the user's search query, retrieves the search results from the MySQL database, and renders the search results using the HTML template.
When a user submits a search query, the data is sent as a request to the search view.
The view reads the search query from the request and fetches the search results from the MySQL database using Pandas. The search results are then converted to a JSON format and stored in the data variable. The search results are then passed to the HTML template context to be rendered in the table. The search results are dynamically rendered in the table using Django template tags.
This view function processes the user's search query, retrieves the search results from the MySQL database, and renders the search results using the HTML template. The search results are then converted to a JSON format and stored in the data variable. The search results are then passed to the HTML template context to be rendered in the table.






                                                    Part 2. Text Classification




1. How training data were collected

Dataset: BBC
All rights, including copyright, in the content of the original articles are owned by the BBC.
•	Consists of 2225 documents from the BBC news website corresponding to stories in five topical areas from 2004-2005.
•	Class Labels: 5 (business, entertainment, politics, sport, Science)
These datasets are made available for non-commercial and research purposes.
Datasets consider citing the publication:
D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006
Snippet of the dataset:
 ![image](https://github.com/Yogesh-653/Information-retrieval/assets/60870157/7c8f580c-9af0-43fa-8f80-0d931b6105be)





2. Which classification method has been used?
There are three classification methods used:
Random Forest Classifier: RandomForestClassifier is used as one of the classifiers to train the model. It is an ensemble learning method that combines multiple decision trees to improve performance and reduce overfitting.
Multinomial Naive Bayes Classifier: MultinomialNB is used as another classifier. It is a variant of the Naive Bayes algorithm specifically designed for multinomially distributed data, which is often used for text classification tasks.
Logistic Regression Classifier: LogisticRegression is used as the third classifier. It is a linear classification algorithm that models the probability of a categorical target variable.
How its performance is measured?
The performance of the classification method is measured using cross-validation. The cross_val_score function from scikit-learn is used to perform cross-validation, which divides the dataset into several folds and evaluates the model's accuracy on different train-test splits. The number of folds used for cross-validation is set to 5 (CV=5) in the code.
Snippet of classification:
 ![image](https://github.com/Yogesh-653/Information-retrieval/assets/60870157/f5672d9d-4248-44ce-a4fa-a360413f6d54)



3. Screenshot and demonstration of its performance using various inputs:
	In this screenshot we are used two type of test case to test the accuracy and robustness of the code.

Test Case – I
I utilised documents with many lines, such as 2-3 lines, for Test Case 1.
For this scenario I’ve used chi-squared test to perform feature selection to find most correlated unigrams and most correlated bigrams.
 ![image](https://github.com/Yogesh-653/Information-retrieval/assets/60870157/1f5cb9b8-d9c8-45e8-bead-24fae0ba35d8)

Result:
 ![image](https://github.com/Yogesh-653/Information-retrieval/assets/60870157/2d3108da-b7ae-453d-a273-c21b6e88d520)

Another case scenario we are finding top unigram and bigram for each news category:
 ![image](https://github.com/Yogesh-653/Information-retrieval/assets/60870157/ba99ab79-7ee9-4c6c-9806-bf05709d7efc)

Result:
 
![image](https://github.com/Yogesh-653/Information-retrieval/assets/60870157/e6903bde-35b6-4fb1-bc0f-0bbf2248f9fa)







Test case – 2
I only utilised documents with a single line for Test Case 2 documents.

 ![image](https://github.com/Yogesh-653/Information-retrieval/assets/60870157/5c5bc4b3-9030-49db-97ba-3c7472e867ad)


Result:
 
![image](https://github.com/Yogesh-653/Information-retrieval/assets/60870157/aeb0e0c8-6d35-47de-a5c5-a2e1ac261a9e)




4. Brief explanation of how it works

Import necessary libraries: The code imports various libraries, including pandas, numpy, matplotlib, nltk, seaborn, and scikit-learn, which are essential for data manipulation, visualization, and machine learning tasks.

Load and preprocess the dataset: The code reads a CSV file named 'bbc.csv' using pandas and assigns numerical category IDs to each news type.

Text preprocessing: The code carries out fundamental text preprocessing operations such stopword removal, stemming, text conversion to lowercase, and punctuation removal. By taking these actions, text representations become more meaningful and noise is minimised.

TF-IDF feature extraction: The preprocessed text is transformed into a TF-IDF (Term Frequency-Inverse Document Frequency) feature matrix by the code using the TfidfVectorizer from scikit-learn. The TF-IDF helps to express text data as numerical characteristics suitable for machine learning by illustrating the significance of each word in a document in relation to a group of texts.

Feature selection: The code performs chi-squared test-based feature selection to identify the most important unigrams and bigrams for each news category. This helps in understanding which words are most correlated with each news category.

Visualization: The code uses t-SNE (t-Distributed Stochastic Neighbor Embedding) for dimensionality reduction to visualize the TF-IDF feature vectors in a 2-dimensional space. The visualization provides insights into how well the data is clustered based on the news categories.

Model training and evaluation: The code uses three machine learning models, namely RandomForestClassifier, MultinomialNB (Naive Bayes), and LogisticRegression, to train and evaluate the performance of each model using cross-validation. The models are evaluated based on their accuracy scores.

Confusion matrix: The code generates a confusion matrix to visualize the performance of the Logistic Regression model on the test data. The confusion matrix shows how well the model predicts the true news categories.

Feature analysis: The code analyzes the top unigrams and bigrams for each news category based on the coefficients of the Logistic Regression model. This provides insights into which words are most important for classifying each news category.

Text classification on new data: Finally, the code performs text classification on a list of new texts using the trained Logistic Regression model. It predicts the news category for each new text and prints the results.


