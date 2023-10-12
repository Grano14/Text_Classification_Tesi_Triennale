import nltk
from nltk.corpus import stopwords
import textblob
from textblob import TextBlob
from textblob import Word
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

#implementazione della funzione per la pulizia del testo grezzo
def text_preprocessing_pipeline(text):
    #lowercasing del testo
    text_cleaned = text.lower()
    #removing puntuaction dal testo grezzo
    segni_punteggiatura = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for carattere in segni_punteggiatura:
        text_cleaned = text_cleaned.replace(carattere, '')
    # Spelling correction con textblob library
    #blobtext = TextBlob(text_cleaned)
    #text_cleaned = blobtext.correct()
    #text_cleaned = text_cleaned.raw
    # Lemmatizzation con textblob library

    blobtext = TextBlob(text_cleaned)
    text_cleaned = blobtext.words.lemmatize()
    # removing stopwords

    stop = set(stopwords.words('english'))
    # Tokenizzazione del testo in parole
    # parole = nltk.word_tokenize(text_cleaned)
    text_cleaned = [parola for parola in text_cleaned if parola.lower() not in stop]
    return text_cleaned

#funzione per il preprocessing dei dati, ottenimento dei dati dal dataset e pulizia dei dati
def preprocessong(data_path):
    #ottenimento dei dati dal file csv, rimozione dei valori nulli
    data_frame = pd.read_csv(data_path)
    data_frame.dropna(subset=['text', 'label'], inplace=True)
    #applicazione della pipeline di pulizia dei dati su ogni riga del dataset
    data_frame['text'] = data_frame['text'].apply(text_preprocessing_pipeline)
    #conversione del contenuto della riga (object: list) in stringa
    data_frame['text'] = data_frame['text'].apply(" ".join)
    return data_frame

data_frame = preprocessong("/home/giuseppe/Scrivania/Università/Tesi/Decision Tree Classifier/Dati/datasetAll.csv")

text_list = data_frame["text"].tolist()
label_list = data_frame["label"].tolist()
#divisione del dataset per il training e il testing
X_train, X_test, y_train, y_test = train_test_split(text_list, label_list, test_size=0.2, random_state=0)



text = "hbdejT;TU.IH5 This is inttroduction  fishes leaves fish leaf"
print(text_preprocessing_pipeline(text))