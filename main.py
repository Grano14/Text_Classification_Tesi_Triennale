import nltk
from nltk.corpus import stopwords
import textblob
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, f1_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from textblob import TextBlob
from textblob import Word
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('punkt')

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

#funzione per la vettorizzazione del testo
def vectorizzation_and_training(X_train, y_train, X_test, func, model):
    # vettorizzazione dei dati utilizzando il modello bag of word
    X_train_bow = func.fit_transform(X_train)
    #vettorizzazione dei dati di test
    X_test_bow = func.transform(X_test)
    #addestramento del modello
    model.fit(X_train_bow, y_train)
    #restituire i valori predetti sui dati di test
    return model.predict(X_test_bow)

#funzione per la stampa delle metriche di valutazione
def print_metrics(pred, y_test):
    print("Matrice di confusione --> ")
    print(confusion_matrix(y_test, pred))
    print("Accuracy --> ")
    print(accuracy_score(y_test, pred))
    print("Precision --> ")
    print(precision_score(y_test, pred, average='macro'))
    print("F1 score --> ")
    print(f1_score(y_test, pred, average='macro'))
    print("Recall --> ")
    print(recall_score(y_test, pred, average='macro'))

#fase di preprocessing del testo
data_frame = preprocessong("/home/giuseppe/Scrivania/Università/Tesi/Decision Tree Classifier/Dati/datasetAll.csv")
#creazione di due liste per i testi e le label presenti nel data_frame
text_list = data_frame["text"].tolist()
label_list = data_frame["label"].tolist()
#divisione del dataset per il training e il testing
X_train, X_test, y_train, y_test = train_test_split(text_list, label_list, test_size=0.2, random_state=0)
#fase di vettorizzazione del testo, addestramento del modello e ottenimento del valori predetti dal modello addestrato sui dati di test
pred = vectorizzation_and_training(X_train, y_train, X_test, CountVectorizer(), DecisionTreeClassifier(random_state=4, max_depth=900))
#stampa delle metriche
print_metrics(pred, y_test)