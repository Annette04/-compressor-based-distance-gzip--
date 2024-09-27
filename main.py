import pandas as pb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import os.path
from saxpy.znorm import znorm
from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import gzip

X = []
Y = []

for root, dirs, files in os.walk('Recording Data'):
    for file in files:
        if file.endswith(".txt") and file != 'Subject codes.txt':
            file_path = os.path.join(root, file)
            match root[-1]:
                case '1' | '2':
                    has_dyslexia = 1
                case _:
                    has_dyslexia = 0
            data = pd.read_csv(file_path, sep='\t', index_col=None, decimal=",")
            data['r_dist'] = (data['RX'].diff() ** 2 + data['RY'].diff() ** 2) ** 0.5
            data['l_dist'] = (data['LX'].diff() ** 2 + data['LY'].diff() ** 2) ** 0.5
            data['r_speed'] = data['r_dist'] / 20 / 1000
            data['l_speed'] = data['l_dist'] / 20 / 1000
            # data['time_stop'] = 0
            # for index, row in data.iterrows():
            #     if row['r_speed'] < 0.00676 and row['l_speed'] < 0.00676:
            #         data.at[index, 'time_stop'] = data.at[index - 1, 'time_stop'] + 20
            # r_sax = [x for x in ts_to_string(znorm(data['r_dist'].to_numpy()[1:]), cuts_for_asize(3))]
            # l_sax = [x for x in ts_to_string(znorm(data['l_dist'].to_numpy()[1:]), cuts_for_asize(3))]
            # X.append(list(zip(data['r_dist'][1:], data['l_dist'][1:])))
            X.append(list(zip(data['r_speed'][1:], data['l_speed'][1:])))
            # X.append(data.iloc[1:])
            # X.append(list(zip( r_sax, l_sax)))
            Y.append(has_dyslexia)
            # plt.plot(data['T'], data['time_stop'])
            # # plt.yticks([x for x in range(0, 800, 80)])
            # plt.title(has_dyslexia)
            # plt.xlabel('Time')
            # plt.ylabel('Fixcation')
            # plt.show()

            # Построить табличку по примеру из статьи
            # Скалярное произведение
            # k fold

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

len(X_test), len(X_train)

k = 23
# Поиграться с k

predicts = []
for x1 in X_test:
    Cx1 = len(gzip.compress(np.array(x1)))
    distance_from_x1 = []
    for x2 in X_train:
        Cx2 = len(gzip.compress(np.array(x2)))
        x1x2 = x1 + x2
        Cx1x2 = len(gzip.compress(np.array(x1x2)))
        ncd = (Cx1x2 - min(Cx1 , Cx2)) / max (Cx1 , Cx2 )
        distance_from_x1.append(ncd)
    
    sorted_idx = np.argsort(np.array(distance_from_x1))
    top_k_class = np.array(Y_train)[sorted_idx[:k]].tolist()
    predict_class = max(set(top_k_class), key = top_k_class.count)
    predicts.append(predict_class)
    
list(zip(predicts, Y_test))

roc_auc = roc_auc_score(Y_test, predicts)
classif = classification_report(Y_test, predicts, output_dict=True)

classif
# 'macro avg': 'f1-score'
