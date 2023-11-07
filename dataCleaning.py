from nltk.corpus import stopwords
from textblob import TextBlob
import pandas as pd
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('punkt')
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
    data_frame.to_csv("/home/giuseppe/Scrivania/Università/Tesi/Decision Tree Classifier/Dati/datasetAllClean.csv")
    return data_frame
