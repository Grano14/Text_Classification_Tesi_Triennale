from matplotlib import pyplot as plt
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, f1_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

#funzione per la vettorizzazione del testo
def vectorizzation_and_training(X_train, y_train, X_test, y_test, func, model):
    # vettorizzazione dei dati utilizzando il modello bag of word
    X_train_bow = func.fit_transform(X_train)
    #vettorizzazione dei dati di test
    X_test_bow = func.transform(X_test)

    sm = SMOTE(random_state=42)
    X_train_bow, y_train = sm.fit_resample(X_train_bow, y_train)
    class_counts = Counter(y_train)


    # Restituisce un dizionario con il conteggio di ciascuna classe
    print(class_counts)
    #addestramento del modello
    model.fit(X_train_bow, y_train)
    #restituire i valori predetti sui dati di test
    return model.predict(X_test_bow)

def testIper(X_train, X_test, y_train, y_test):
    print('MultinomialNB')
    params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    best_param = 0
    best_value = 0
    for param in params:
        # fase di vettorizzazione del testo, addestramento del modello e ottenimento del valori predetti dal modello addestrato sui dati di test
        pred = vectorizzation_and_training(X_train, y_train, X_test, y_test, CountVectorizer(), MultinomialNB(alpha=param))
        # stampa delle metriche
        # print_metrics(pred, y_test)
        if best_value < accuracy_score(y_test, pred):
            best_value = accuracy_score(y_test, pred)
            best_param = param
        print(f1_score(y_test, pred, average='micro'))
    print('best param = ' + str(best_param) + 'best value = ' + str(best_value))
    print(confusion_matrix(y_test, pred))
    confusion = confusion_matrix(y_test, pred)
    import seaborn as sns

    # Creare una heatmap per la matrice di confusione
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.legend()
    plt.show()

    print('LinearSVC')
    params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    best_param = 0
    best_value = 0
    for param in params:
        # fase di vettorizzazione del testo, addestramento del modello e ottenimento del valori predetti dal modello addestrato sui dati di test
        pred = vectorizzation_and_training(X_train, y_train, X_test, y_test, CountVectorizer(), LinearSVC(dual=True, C=param, max_iter=100000))
        # stampa delle metriche
        # print_metrics(pred, y_test)
        if best_value < accuracy_score(y_test, pred):
            best_value = accuracy_score(y_test, pred)
            best_param = param
        print(f1_score(y_test, pred, average='micro'))
    print('best param = ' + str(best_param) + 'best value = ' + str(best_value))
    print(confusion_matrix(y_test, pred))

    print('tree')
    params = [["gini", "best"], ["gini", "random"], ["entropy", "best"], ["entropy", "random"]]
    best_param = 0
    best_value = 0
    for param in params:
        # fase di vettorizzazione del testo, addestramento del modello e ottenimento del valori predetti dal modello addestrato sui dati di test
        pred = vectorizzation_and_training(X_train, y_train, X_test, y_test, CountVectorizer(), DecisionTreeClassifier(criterion=param[0], splitter=param[1]))
        # stampa delle metriche
        # print_metrics(pred, y_test)
        if best_value < accuracy_score(y_test, pred):
            best_value = accuracy_score(y_test, pred)
            best_param = param
        print(f1_score(y_test, pred, average='micro'))
    print('best param = ' + str(best_param) + 'best value = ' + str(best_value))
