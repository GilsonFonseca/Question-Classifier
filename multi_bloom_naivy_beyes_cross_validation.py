import os
import tensorflow as tf
import files_ms_client
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import matthews_corrcoef
import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

# GPU

device_name = tf.test.gpu_device_name()
if device_name == '/device:GPU:0':
  print('GPU encontrada em: {}'.format(device_name))
else:
  print('GPU não encontrada')

# Set enviroment configs up. Default values can be changed by altering
# the second argument in each "get" call
FILES_SERVER = os.environ.get("FILES_SERVER", "200.17.70.211:10162")

def myCode(msg):
    FILE = 'basebloom.xlsx'
    FILE_OUTPUT = 'questions.xlsx'

    files_ms_client.download(msg["file"]["name"], FILE_OUTPUT, url="http://" + FILES_SERVER)

  # Basebloom
    df = pd.read_excel(FILE)
    df.sample(frac=1)

  # Base Questions
    dq = pd.read_excel(FILE)
    dq.sample(frac=1)

    # Remove os dados N/A da base, assim como tabelas não usadas
    df = df.dropna()

    # Recebe df
    dataset = df.copy()

    num_classes = len(df["Level"].value_counts())

    dataset['category_id'] = df['Level'].factorize()[0]
    # category_id_df = dataset[['Level', 'category_id']].drop_duplicates().sort_values('category_id')
    # category_to_id = dict(category_id_df.values)
    # id_to_category = dict(category_id_df[['category_id', 'Level']].values)

    dataset['Labels'] = dataset['Level'].map({'Knowledge': 0,
                                                'Comprehension': 1,
                                                'Application': 2,
                                                'Analysis': 3,
                                                'Synthesis': 4,
                                                'Evalution': 5})

    dataset['Question'] = dataset['Question'].str.lower()

    dataset.sample(10)

    datasetTest = dataset.groupby('Labels').sample(n=17)
    datasetTest.to_csv('datasetTest.csv', index=False)
    dataset = dataset.drop_duplicates(subset="Question")
    for row, data in enumerate(dataset.values):
        for dup in datasetTest.values:
            if((data == dup).all()):
                dataset = dataset[dataset.Question != data[0]]

    # Recebe a lista com as sentenças e os labels
    sentences = dataset.Question.values
    labels = dataset.Labels.values

    X = sentences
    y = labels
    skf = StratifiedKFold(n_splits=10, shuffle = True)
    skf.get_n_splits(X, y)

    resultArray = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]

        vect = CountVectorizer(min_df=1, encoding='latin-1', ngram_range=(1, 3), stop_words='english')
        X_train_dtm = vect.fit_transform(X_train)
        X_val_dtm = vect.transform(X_val)

        mnb = MultinomialNB()
        mnb.fit(X_train_dtm, y_train)
        resultArray.append(mnb.score(X_val_dtm, y_val))

    resultArray = np.array(resultArray)
    print(f"Cross {resultArray}")
    print(f"Média {np.mean(resultArray)}")
    print(f"Desvio Padrão {np.std(resultArray)}")

    sentencesTest = datasetTest.Question.values
    y_true_man = datasetTest.Labels.values
    x_test = vect.transform(sentencesTest)
    y_test = mnb.predict(x_test)
    print(f"Micro {precision_recall_fscore_support(y_true_man, y_test, average='micro')}") # y_true, são as classificações corretas da base de teste, e y_test os resultados da predição do MNB
    print(f"Macro {precision_recall_fscore_support(y_true_man, y_test, average='macro')}")
    print(f"Weighted {precision_recall_fscore_support(y_true_man, y_test, average='weighted')}")

    cm = confusion_matrix(y_true_man, y_test)
    mcc = matthews_corrcoef(y_true_man, y_test)
    av = precision_recall_fscore_support(y_true_man, y_test, average='micro')

    print(f"MCC {mcc}")
    print(metrics.classification_report(y_true_man, y_test, labels=[0,1,2,3,4,5]))
    print(av)
    df_cm = pd.DataFrame(cm, range(num_classes), range(num_classes))
    plt.figure(figsize=(11,8))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

    graph = plt.savefig("graph.png")

    msg["qc-image"] = files_ms_client.upload("graph.png", url="http://" + FILES_SERVER)

    return msg