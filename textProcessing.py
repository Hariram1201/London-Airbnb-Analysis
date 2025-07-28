import pandas as pd #Library used for data manipulation and analysis 
import matplotlib.pyplot as plt #Library used to create static, interactive and animated visualisations

import re #Library provides support for working with regular expressions — powerful tools for searching, matching, and manipulating text using patterns

import nltk #Library to import the entire NLTK library 
from nltk.tokenize import word_tokenize #Splits text into individual words and punctuation
from nltk.corpus import stopwords #Gives access to a list of commo0n words to ignore 
from nltk.stem import WordNetLemmatizer #Reduces words to their base form
from nltk import pos_tag #Gives part of speech tagging - labels each token with its grammatical role

from sentence_transformers import SentenceTransformer #Library to load pre-trained models to convert text into dense vector embeddings
from sklearn.metrics.pairwise import cosine_similarity #Library to use cosine similarity to determine how similar two pieces of text are 

from deep_translator import GoogleTranslator #Library to translate text between languages using Google Translate

from wordcloud import WordCloud #Library to create a word cloud of the most frequently used words
from collections import Counter #Imports a class to count hashable objects efficiently, useful for frequency analysis

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer #Imports a tool to analyze the sentiment polarity of text (positive, negative, neutral)

from gensim import corpora, models #Imports modules for topic modeling and creating document-term matrices

# Make sure nltk resources are properly downloaded
nltk.download('punkt_tab')  # Tokeniser to split text into tokens (words)
nltk.download('stopwords')  # Stopwords list to remove common non-informative words
nltk.download('wordnet') # WordNet for lemmatisation (base form of words)
nltk.download('averaged_perceptron_tagger_eng') # Part-of-Speech tagging to know word types

# Create an instance of the lemmatiser
lemmatiser = WordNetLemmatizer()

# Load a pre-trained model to convert full sentences into 384 dimensional dense vectors
model = SentenceTransformer('all-MiniLM-L6-v2')

analyzer = SentimentIntensityAnalyzer()

# Function to map POS tag from Treebank format to WordNet format
def get_wordnet_pos(treebank_tag):
    """Converts Treebank POS tags to WordNet POS tags"""
    if treebank_tag.startswith('J'):
        return 'a'  # Adjective
    elif treebank_tag.startswith('V'):
        return 'v'  # Verb
    elif treebank_tag.startswith('N'):
        return 'n'  # Noun
    elif treebank_tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return 'n'  # Default to noun if unsure
    
def cleanText(dataset, targetColumn, tokenisedCol):

    """
    Clean and normalize text data by lowercasing and removing punctuation/numbers.

    Parameters:
    -----------
    dataset : pandas.DataFrame
        The dataset containing the text column to be cleaned.
    targetColumn : str
        Column name in the dataset containing raw text.
    tokenisedCol : str
        Name of the new column to store the cleaned text.

    Returns:
    --------
    dataset : pandas.DataFrame
        The updated DataFrame with the cleaned text in a new column.
    """

    #Insert new column for cleaned text right after the target column
    colIndex = dataset.columns.get_loc(targetColumn)
    dataset.insert(colIndex + 1, tokenisedCol, None)

    #Iterate through each row to process the text
    for index, row in dataset.iterrows():

        #Convert to string and handle NaN
        text = row[targetColumn]
        if pd.isna(text):
            text = ""
        else:
            text = str(text)

        #Convert to lowercase
        text = text.lower()

        #Remove all non-letter characters (punctuation, digits, etc.)
        text = re.sub(r'[^a-z\s]', '', text)

        #Replace multiple spaces with a single space and strip leading/trailing spaces
        text = re.sub(r'\s+', ' ', text).strip()  

        #Store cleaned text in the new column
        dataset.at[index, tokenisedCol] = text

    return dataset

def tokenise(dataset, tokenisedCol, customStopwords):

    """
    Tokenise and lemmatise text data while removing stopwords and irrelevant tokens.

    Parameters:
    -----------
    dataset : pandas.DataFrame
        The dataset containing the pre-cleaned text column.
    tokenisedCol : str
        Name of the column containing cleaned text to be tokenised and lemmatised.
    customStopwords : list
        List of additional stopwords to exclude during tokenisation.

    Returns:
    --------
    dataset : pandas.DataFrame
        The updated DataFrame with tokenised and lemmatised text in the specified column.
    """

    #Combine built-in English and French stopwords with custom stopwords
    stopWords = set(stopwords.words('english')).union(set(stopwords.words('french')))
    stopWords.update(customStopwords)

    #Iterate over each row in the dataset
    for index, row in dataset.iterrows():

        #Extract the cleaned text for processing
        text = row[tokenisedCol]

        #Tokenise the text into words
        tokens = word_tokenize(text)

        #List to store filtered and lemmatised tokens
        filteredTokens = []

        #Get part-of-speech tags for each token
        posTags = pos_tag(tokens)

        #Process each token individually
        for word, tag in posTags:
            # Remove stopwords and words with length 1 (e.g., 'a', 'I')
            if word != "br" and word != "n't" and len(word) > 1 and word.lower() not in stopWords:
                wordnetPos = get_wordnet_pos(tag)  # Get WordNet POS tag
                lemmatisedWord = lemmatiser.lemmatize(word, pos=wordnetPos)  # Lemmatize based on POS
                filteredTokens.append(lemmatisedWord)
                if word == "u":
                    print("u found")
        
        #Store the list of tokens back in the specified column
        dataset.at[index, tokenisedCol] = filteredTokens 

    return dataset

def get_similarityMatrix(sentences):

    """
    Generate sentence embeddings and compute the cosine similarity matrix.

    Parameters:
    -----------
    sentences : list of str
        A list containing the sentences to compare.

    Returns:
    --------
    similarityMatrix : numpy.ndarray
        A 2D array representing cosine similarity scores between sentence embeddings.
    """

    #Generate sentence embeddings using a pre-trained model
    embeddings = model.encode(sentences)

    #Compute pairwise cosine similarity between the embeddings
    similarityMatrix = cosine_similarity(embeddings)

    return similarityMatrix

def translateToEnglish(text):

    """
    Translates input text to English using the GoogleTranslator library.

    Parameters:
    -----------
    text : str
        The input string in any language to be translated.

    Returns:
    --------
    str
        The translated English string. If translation fails due to an error,
        the original input string is returned as a fallback.
    """

    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text  # fallback if translation fails
    
def wordCloud(filteredWords, title):

    """
    Create and display a word cloud from a list or series of words.

    Parameters:
    -----------
    filteredWords : iterable
        Column or list containing words, where each item can be a string or a list of words.
    title : str
        Title for the generated word cloud plot.
    """

    #Flatten the list: combine all words from sublists or single strings into one list
    flattened = []

    for item in filteredWords:
        if isinstance(item, list):
            flattened.extend(item)  # Add all words from the sublist
        else:
            flattened.append(item)  # Add the single word

    #Convert list of words to a single string
    cleanText = ' '.join(flattened)

    #Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleanText)

    #Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

def wordFreq(filteredWords, title):

    """
    Create a horizontal bar plot of the top 20 most frequent words.

    Parameters:
    -----------
    filteredWords : iterable
        Column or list containing words, where each item can be a string or a list of words.
    title : str
        Title for the frequency plot.
    """

    # Flatten the list: combine all words from sublists or single strings into one list
    allWords = []

    for item in filteredWords:
        if isinstance(item, list):
            allWords.extend(item)  
        else:
            allWords.append(item)  

    #Count the occurrences of each word
    wordCounts = Counter(allWords)

    #Extract the 20 most common words and their counts
    mostCommon = wordCounts.most_common(20)
    words = [item[0] for item in mostCommon]
    counts = [item[1] for item in mostCommon]

    #Plot the horizontal bar chart
    plt.figure(figsize=(10,6))
    plt.barh(words, counts, color='lightblue')
    plt.xlabel('Frequency')
    plt.title(title)
    plt.gca().invert_yaxis()  
    plt.show()

def vaderSentiment(text):

    """
    Computes the sentiment score of a text using VADER (Valence Aware Dictionary for Sentiment Reasoning).

    Parameters:
    -----------
    text : str
        The input string (e.g., a review) to analyze for sentiment.

    Returns:
    --------
    score['compound'] : float
        A compound sentiment score ranging from -1 (most negative) to +1 (most positive).
        If the input is missing or not a string, returns 0 as a neutral fallback.
    """
    if pd.isna(text) or not isinstance(text, str):
        return 0
    score = analyzer.polarity_scores(text)
    return score['compound']  

def lda(df, columnName):

    """
    Performs Latent Dirichlet Allocation (LDA) topic modeling on a column of tokenized texts,
    assigns the most likely topic to each review, and adds that topic description back to the DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the review data.

    columnName : str
        The name of the column in df that contains tokenized review texts (lists of words).

    Returns:
    --------
    df : pandas.DataFrame
        The original DataFrame with a new column 'top_topic' containing the descriptive string of the
        most relevant topic assigned to each review.

    topics : list of tuples
        A list of tuples where each tuple contains a topic number and its top descriptive words.

    topTopics : list
        A list of the top topic numbers assigned to each review (aligned with df).
    """

    texts = df[columnName]

    #Create dictionary and corpus
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    #Train LDA model
    ldaModel = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)

    # Get the top topic for each review
    topTopics = []
    for doc in corpus:
        topicDistribution = ldaModel.get_document_topics(doc)
        if topicDistribution:
            # Pick the topic with the highest probability for this document
            topTopic = max(topicDistribution, key=lambda x: x[1])[0]
        else:
            topTopic = None
        topTopics.append(topTopic)

    # Create a dictionary mapping topic number to descriptive top words
    topics = ldaModel.print_topics(num_words=5)

    # Add the topic description labels to the DataFrame (align indexes)
    df.loc[texts.index, 'top_topic'] = topTopics

    # Print topics for reference
    for i, topic in topics:
        print(f"Topic #{i + 1}: {topic}")

    return df, topics, topTopics