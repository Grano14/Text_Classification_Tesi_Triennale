import nltk
from nltk.corpus import stopwords
import textblob
from textblob import TextBlob
from textblob import Word
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
#implementazione della funzione per la pulizia del testo grezzo
def text_preprocessing_pipeline(text):
    #lowercasing del testo
    text_cleaned = text.lower()
    #removing puntuaction dal testo grezzo
    segni_punteggiatura = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for carattere in segni_punteggiatura:
        text_cleaned = text_cleaned.replace(carattere, '')
    #Spelling correction con textblob library
    blobtext = TextBlob(text_cleaned)
    text_cleaned = blobtext.correct()
    text_cleaned = text_cleaned.raw
    # Lemmatizzation con textblob library
    nltk.download('wordnet')
    blobtext = TextBlob(text_cleaned)
    text_cleaned = blobtext.words.lemmatize()
    # removing stopwords
    nltk.download('stopwords')
    nltk.download('punkt')
    stop = set(stopwords.words('english'))
    # Tokenizzazione del testo in parole
    #parole = nltk.word_tokenize(text_cleaned)
    text_cleaned = [parola for parola in text_cleaned if parola.lower() not in stop]
    return text_cleaned

#ottenimento dei dati dal file csv, rimozione dei valori nulli, creazione di due liste relative ai testi e alle label
data_frame = pd.read_csv("/home/giuseppe/Scrivania/Università/Tesi/Decision Tree Classifier/Dati/datasetAll.csv")
data_frame.dropna(subset=['text', 'label'], inplace=True)
text_list = data_frame["text"].tolist()
label_list = data_frame["label"].tolist()
#divisione del dataset per il training e il testing
X_train, X_test, y_train, y_test = train_test_split(text_list, label_list, test_size=0.2, random_state=0)

text = "hbdejT;TU.IH5 This is inttroduction  fishes leaves fish leaf"
print(text_preprocessing_pipeline(text))