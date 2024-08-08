import PyPDF2
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import unicodedata
from contractions import contractions_dict
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import spacy

### get text from pdf
def get_text_from_pdf(countries_legend, country, pg_from, pg_to, dimension, save_txt = False):
    pg_from = pg_from - 1
    pg_to = pg_to - 1
    doc_name = countries_legend.loc[countries_legend['country'] == country, 'doc_name'].iloc[0]
    pdf_file = open("Energy National Plans/" + doc_name + ".pdf", "rb")
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    all_text = ''
    for i in range(pg_to - pg_from):
        doc = pdf_reader.pages[pg_from + i]
        all_text = all_text + doc.extract_text()

    if save_txt:
                txt = open('output/' + dimension + '_txt/' + country + ".txt", 'w', encoding="utf-8")
                txt.write(all_text)
                txt.close()
    return all_text

    

### tokenize text
def tokenize_text(text):
    # 1 token per sentence
    sentences = sent_tokenize(text)
    word_tokens = [word_tokenize(sentence) for sentence in sentences]

    # 1 token per word
    word_tokens = word_tokenize(text)

    return word_tokens

### remove accents from text
def remove_accents(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

# expand contractions (e.g. you're --> you are)
def expand_contractions(text, contraction_mapping=contractions_dict):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), flags=re.IGNORECASE|re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match) else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
    
    try :
        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
    except TypeError:
        expanded_text = text

    return expanded_text

# remove special characters
def remove_special_char(text, remove_digits = True, keep_years = True):
    text = re.sub(r'[^\w\s]','', text)
    text = re.sub(r'\b\d{1,3}\b|\b\d{5,}\b', '', text) if remove_digits else text
    return text


def lemmatize_text(text):
    sentences = sent_tokenize(text)
    lemmatized_sentences = []

    nlp = spacy.load('en_core_web_sm')
    for sentence in sentences:
        lemmatized_sentences.append(' '.join([nlp(words)[0].lemma_ for words in word_tokenize(sentence)]).lower())

    return ' '.join(lemmatized_sentences)

# remove stop words
def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)

    words_1 = [word for word in words if word not in stop_words]
    return ' '.join(words_1)

############ Global function ############

def normalize_corpus(corpus, country, dimension, contractions = True, accents = True, special_char = True, lemmatize = True, stop_words = True, save_output = False, printt = True):
        len_0 = len(word_tokenize(corpus))
        if contractions:
            corpus = expand_contractions(corpus)
        if accents:
            corpus = remove_accents(corpus)
        if special_char:
            corpus = remove_special_char(corpus)
        if lemmatize:
            corpus = lemmatize_text(corpus)
        if stop_words:
            corpus = remove_stop_words(corpus)
        len_f = len(word_tokenize(corpus))

        if printt:
            print("{} --> {} words, reduction of {}%".format(len_0, len_f, round(len_f/len_0*100)))
        
        if save_output:
                txt = open('output/' + dimension + '_norm/' + country + ".txt", 'w')
                txt.write(corpus)
                txt.close()

        return corpus

#########################################

def crete_BoW_matrix(all_corpus, countries, N_gram):
    labels, documents = zip(*all_corpus)
    if N_gram == 1:
        cv = CountVectorizer(max_features=20, min_df=0., max_df=1.)
    else:
        cv = CountVectorizer(ngram_range=(N_gram,N_gram), max_features=20, min_df=0., max_df=1.)
    cv_matrix = cv.fit_transform(documents)
    feature_names = cv.get_feature_names_out()
    cv_df = pd.DataFrame(cv_matrix.toarray(), columns=feature_names)
    cv_df.index = countries

    return cv_df

def create_TF_IDF_matrix(all_corpus, countries):
    labels, documents = zip(*all_corpus)
    tfidf_vectorizer = TfidfVectorizer(max_features = 20, stop_words = 'english', norm = 'l2')
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(np.round(tfidf_matrix.toarray(), 2), columns=feature_names)
    tfidf_df.index = countries
    
    return tfidf_df

def create_heatmap(df, title):
    # sns.set()
    sns.set(font_scale=1.4)
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, cmap='coolwarm', fmt='g')
    plt.title(str(title))
    # plt.xlabel("Words")
    plt.show()




def normalize_sentence(sentence):
    # stop words
    sentence = remove_stop_words(sentence)

    return sentence







