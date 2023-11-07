import os

from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from matplotlib import pyplot as plt

import dataCleaning
import  testIperparameters
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, f1_score, recall_score, \
    ConfusionMatrixDisplay
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import seaborn as sns

#fase di preprocessing del testo
percorso_file = "/home/giuseppe/Scrivania/Università/Tesi/Decision Tree Classifier/Dati/datasetAllClean.csv"
if os.path.exists(percorso_file):
    data_frame = pd.read_csv(percorso_file)
else:
    data_frame = dataCleaning.preprocessong("/home/giuseppe/Scrivania/Università/Tesi/Decision Tree Classifier/Dati/datasetAll.csv")
#creazione di due liste per i testi e le label presenti nel data_frame
text_list = data_frame["text"].tolist()
label_list = data_frame["label"].tolist()

#divisione del dataset per il training e il testing
X_train, X_test, y_train, y_test = train_test_split(text_list, label_list, test_size=0.2, random_state=0)

#testIperparameters.testIper(X_train, X_test, y_train, y_test)

# vettorizzazione dei dati utilizzando il modello bag of word
func = CountVectorizer()
X_train_bow = func.fit_transform(X_train)
#vettorizzazione dei dati di test
X_test_bow = func.transform(X_test)

from collections import Counter
class_counts = Counter(y_train)
print(class_counts)
#oversampling dei dati
sm = SMOTE(random_state=42)
X_train_bow, y_train = sm.fit_resample(X_train_bow, y_train)
class_counts = Counter(y_train)
print(class_counts)
#addestramento del modello
model = MultinomialNB(alpha=0.2)
model.fit(X_train_bow, y_train)
#restituire i valori predetti sui dati di test
pred = model.predict(X_test_bow)
print(f1_score(y_test, pred, average='micro'))
print(confusion_matrix(y_test, pred))
confusion = confusion_matrix(y_test, pred)
# Creare una heatmap per la matrice di confusione
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

cm = confusion_matrix(y_test, pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()