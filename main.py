import nltk
from nltk.corpus import stopwords
import textblob
from textblob import TextBlob
from textblob import Word
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
    text_cleaned = text_cleaned.raw
    # removing stopwords
    nltk.download('stopwords')
    nltk.download('punkt')
    stop = set(stopwords.words('english'))
    # Tokenizzazione del testo in parole
    parole = nltk.word_tokenize(text_cleaned)
    text_cleaned = [parola for parola in parole if parola.lower() not in stop]
    return text_cleaned


text = "hbdejT;TU.IH5 This is inttroduction  fishes leaves fish leaf"
print(text_preprocessing_pipeline(text))