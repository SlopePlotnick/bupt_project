import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn import preprocessing

# Import required libraries
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

mult = 5

# def load_file(path):
#     data = pd.read_csv(path, sep=',')
#
#     is_benign = data[' Label'] == 'BENIGN'
#     flows_ok = data[is_benign]
#     flows_ddos_full = data[~is_benign]
#
#     sizeDownSample = len(flows_ok) * mult  # tamanho do set final de dados anomalos
#
#     # downsample majority
#     if (len(flows_ok) * mult) < (len(flows_ddos_full)):
#         flows_ddos_reduced = resample(flows_ddos_full,
#                                       replace=False,  # sample without replacement
#                                       n_samples=sizeDownSample,  # match minority n
#                                       random_state=27)  # reproducible results
#     else:
#         flows_ddos_reduced = flows_ddos_full
#
#     return flows_ok, flows_ddos_reduced
#
#
# def load_huge_file(path):
#     df_chunk = pd.read_csv(path, chunksize=500000)
#
#     chunk_list_ok = []  # append each chunk df here
#     chunk_list_ddos = []
#
#     # Each chunk is in df format
#     for chunk in df_chunk:
#         # perform data filtering
#         is_benign = chunk[' Label'] == 'BENIGN'
#         flows_ok = chunk[is_benign]
#         flows_ddos_full = chunk[~is_benign]
#
#         if (len(flows_ok) * mult) < (len(flows_ddos_full)):
#             sizeDownSample = len(flows_ok) * mult  # tamanho do set final de dados anomalos
#
#             # downsample majority
#             flows_ddos_reduced = resample(flows_ddos_full,
#                                           replace=False,  # sample without replacement
#                                           n_samples=sizeDownSample,  # match minority n
#                                           random_state=27)  # reproducible results
#         else:
#             flows_ddos_reduced = flows_ddos_full
#
#         # Once the data filtering is done, append the chunk to list
#         chunk_list_ok.append(flows_ok)
#         chunk_list_ddos.append(flows_ddos_reduced)
#
#     # concat the list into dataframe
#     flows_ok = pd.concat(chunk_list_ok)
#     flows_ddos = pd.concat(chunk_list_ddos)
#
#     return flows_ok, flows_ddos
#
# # file 1
# flows_ok, flows_ddos = load_huge_file('cicddos2019/01-12/TFTP.csv')
# print('file 1 loaded')
#
# # file 2
# a,b = load_file('cicddos2019/01-12/DrDoS_LDAP.csv')
# flows_ok = flows_ok.append(a,ignore_index=True)
# flows_ddos = flows_ddos.append(b,ignore_index=True)
# print('file 2 loaded')
#
# # file 3
# a,b = load_file('cicddos2019/01-12/DrDoS_MSSQL.csv')
# flows_ok = flows_ok.append(a,ignore_index=True)
# flows_ddos = flows_ddos.append(b,ignore_index=True)
# print('file 3 loaded')
#
# # file 4
# a,b = load_file('cicddos2019/01-12/DrDoS_NetBIOS.csv')
# flows_ok = flows_ok.append(a,ignore_index=True)
# flows_ddos = flows_ddos.append(b,ignore_index=True)
# print('file 4 loaded')
#
# # file 5
# a,b = load_file('cicddos2019/01-12/DrDoS_NTP.csv')
# flows_ok = flows_ok.append(a,ignore_index=True)
# flows_ddos = flows_ddos.append(b,ignore_index=True)
# print('file 5 loaded')
#
# # file 6
# a,b = load_file('cicddos2019/01-12/DrDoS_SNMP.csv')
# flows_ok = flows_ok.append(a,ignore_index=True)
# flows_ddos = flows_ddos.append(b,ignore_index=True)
# print('file 6 loaded')
#
# # file 7
# a,b = load_file('cicddos2019/01-12/DrDoS_SSDP.csv')
# flows_ok = flows_ok.append(a,ignore_index=True)
# flows_ddos = flows_ddos.append(b,ignore_index=True)
# print('file 7 loaded')
#
# # file 8
# a,b = load_file('cicddos2019/01-12/DrDoS_UDP.csv')
# flows_ok = flows_ok.append(a,ignore_index=True)
# flows_ddos = flows_ddos.append(b,ignore_index=True)
# print('file 8 loaded')
#
# # file 9
# a,b = load_file('cicddos2019/01-12/Syn.csv')
# flows_ok = flows_ok.append(a,ignore_index=True)
# flows_ddos = flows_ddos.append(b,ignore_index=True)
# print('file 9 loaded')
#
# # file 10
# a,b = load_file('cicddos2019/01-12/DrDoS_DNS.csv')
# flows_ok = flows_ok.append(a,ignore_index=True)
# flows_ddos = flows_ddos.append(b,ignore_index=True)
# print('file 10 loaded')
#
# # file 11
# a,b = load_file('cicddos2019/01-12/UDPLag.csv')
# flows_ok = flows_ok.append(a,ignore_index=True)
# flows_ddos = flows_ddos.append(b,ignore_index=True)
# print('file 11 loaded')
#
# del a,b
#
# samples = flows_ok.append(flows_ddos,ignore_index=True)
# samples.to_csv(r'cicddos2019/01-12/export_dataframe.csv', index = None, header=True)
#
# del flows_ddos, flows_ok
#
# # file 1
# flows_ok, flows_ddos = load_file('cicddos2019/03-11/LDAP.csv')
# print('file 1 loaded')
#
# # file 2
# a, b = load_file('cicddos2019/03-11/MSSQL.csv')
# flows_ok = flows_ok.append(a, ignore_index=True)
# flows_ddos = flows_ddos.append(b, ignore_index=True)
# print('file 2 loaded')
#
# # file 3
# a, b = load_file('cicddos2019/03-11/NetBIOS.csv')
# flows_ok = flows_ok.append(a, ignore_index=True)
# flows_ddos = flows_ddos.append(b, ignore_index=True)
# print('file 3 loaded')
#
# # file 4
# a, b = load_file('cicddos2019/03-11/Portmap.csv')
# flows_ok = flows_ok.append(a, ignore_index=True)
# flows_ddos = flows_ddos.append(b, ignore_index=True)
# print('file 4 loaded')
#
# # file 5
# a, b = load_file('cicddos2019/03-11/Syn.csv')
# flows_ok = flows_ok.append(a, ignore_index=True)
# flows_ddos = flows_ddos.append(b, ignore_index=True)
# print('file 5 loaded')
# '''
# # following files won't load**
# # file 6
#
# a,b = load_file('cicddos2019/03-11/UDP.csv')
# flows_ok = flows_ok.append(a,ignore_index=True)
# flows_ddos = flows_ddos.append(b,ignore_index=True)
# print('file 6 loaded')
#
# # file 7
# a,b = load_file('cicddos2019/03-11/UDPLag.csv')
# flows_ok = flows_ok.append(a,ignore_index=True)
# flows_ddos = flows_ddos.append(b,ignore_index=True)
# print('file 7 loaded')
# '''
# tests = flows_ok.append(flows_ddos, ignore_index=True)
# tests.to_csv(r'cicddos2019/01-12/export_tests.csv', index=None, header=True)
#
# del flows_ddos, flows_ok, a, b
#
# # training data
# samples = pd.read_csv('cicddos2019/01-12/export_dataframe.csv', sep=',')
#
# def string2numeric_hash(text):
#     import hashlib
#     return int(hashlib.md5(text).hexdigest()[:8], 16)
#
# # Flows Packet/s e Bytes/s - Replace infinity by 0
# samples = samples.replace('Infinity','0')
# samples = samples.replace(np.inf,0)
# #samples = samples.replace('nan','0')
# samples[' Flow Packets/s'] = pd.to_numeric(samples[' Flow Packets/s'])
#
# samples['Flow Bytes/s'] = samples['Flow Bytes/s'].fillna(0)
# samples['Flow Bytes/s'] = pd.to_numeric(samples['Flow Bytes/s'])
#
#
# #Label
# samples[' Label'] = samples[' Label'].replace('BENIGN',0)
# samples[' Label'] = samples[' Label'].replace('DrDoS_DNS',1)
# samples[' Label'] = samples[' Label'].replace('DrDoS_LDAP',1)
# samples[' Label'] = samples[' Label'].replace('DrDoS_MSSQL',1)
# samples[' Label'] = samples[' Label'].replace('DrDoS_NTP',1)
# samples[' Label'] = samples[' Label'].replace('DrDoS_NetBIOS',1)
# samples[' Label'] = samples[' Label'].replace('DrDoS_SNMP',1)
# samples[' Label'] = samples[' Label'].replace('DrDoS_SSDP',1)
# samples[' Label'] = samples[' Label'].replace('DrDoS_UDP',1)
# samples[' Label'] = samples[' Label'].replace('Syn',1)
# samples[' Label'] = samples[' Label'].replace('TFTP',1)
# samples[' Label'] = samples[' Label'].replace('UDP-lag',1)
# samples[' Label'] = samples[' Label'].replace('WebDDoS',1)
#
# #Timestamp - Drop day, then convert hour, minute and seconds to hashing
# colunaTime = pd.DataFrame(samples[' Timestamp'].str.split(' ',1).tolist(), columns = ['dia','horas'])
# colunaTime = pd.DataFrame(colunaTime['horas'].str.split('.',1).tolist(),columns = ['horas','milisec'])
# stringHoras = pd.DataFrame(colunaTime['horas'].str.encode('utf-8'))
# samples[' Timestamp'] = pd.DataFrame(stringHoras['horas'].apply(string2numeric_hash))#colunaTime['horas']
# del colunaTime,stringHoras
#
#
# # flowID - IP origem - IP destino - Simillar HTTP -> Drop (individual flow analysis)
# del samples[' Source IP']
# del samples[' Destination IP']
# del samples['Flow ID']
# del samples['SimillarHTTP']
# del samples['Unnamed: 0']
#
# samples.to_csv(r'cicddos2019/01-12/export_dataframe_proc.csv', index = None, header=True)
# print('Training data processed')
#
# # test data
# tests = pd.read_csv('cicddos2019/01-12/export_tests.csv', sep=',')
#
# def string2numeric_hash(text):
#     import hashlib
#     return int(hashlib.md5(text).hexdigest()[:8], 16)
#
# # Flows Packet/s e Bytes/s - Change infinity by 0
# tests = tests.replace('Infinity','0')
# tests = tests.replace(np.inf,0)
# #amostras = amostras.replace('nan','0')
# tests[' Flow Packets/s'] = pd.to_numeric(tests[' Flow Packets/s'])
#
# tests['Flow Bytes/s'] = tests['Flow Bytes/s'].fillna(0)
# tests['Flow Bytes/s'] = pd.to_numeric(tests['Flow Bytes/s'])
#
#
# #Label
# tests[' Label'] = tests[' Label'].replace('BENIGN',0)
# tests[' Label'] = tests[' Label'].replace('LDAP',1)
# tests[' Label'] = tests[' Label'].replace('NetBIOS',1)
# tests[' Label'] = tests[' Label'].replace('MSSQL',1)
# tests[' Label'] = tests[' Label'].replace('Portmap',1)
# tests[' Label'] = tests[' Label'].replace('Syn',1)
# #tests[' Label'] = tests[' Label'].replace('DrDoS_SNMP',1)
# #tests[' Label'] = tests[' Label'].replace('DrDoS_SSDP',1)
#
# #Timestamp - Drop day, then convert hour, minute and seconds to hashing
# colunaTime = pd.DataFrame(tests[' Timestamp'].str.split(' ',1).tolist(), columns = ['dia','horas'])
# colunaTime = pd.DataFrame(colunaTime['horas'].str.split('.',1).tolist(),columns = ['horas','milisec'])
# stringHoras = pd.DataFrame(colunaTime['horas'].str.encode('utf-8'))
# tests[' Timestamp'] = pd.DataFrame(stringHoras['horas'].apply(string2numeric_hash))#colunaTime['horas']
# del colunaTime,stringHoras
#
# # flowID - IP origem - IP destino - Simillar HTTP -> Deletar (analise fluxo a fluxo)
# del tests[' Source IP']
# del tests[' Destination IP']
# del tests['Flow ID']
# del tests['SimillarHTTP']
# del tests['Unnamed: 0']
#
# tests.to_csv(r'cicddos2019/01-12/export_tests_proc.csv', index = None, header=True)
# print('Test data processed')

def SVM():
    return SVC(kernel='linear')

def LR():
    return LogisticRegression()

def kNN():
    return KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

def RF():
    return RandomForestClassifier()

def DT():
    return DecisionTreeClassifier()

def NB():
    return GaussianNB()


def train_test(samples):
    # Import `train_test_split` from `sklearn.model_selection`
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import SelectKBest, f_classif
    import numpy as np

    # Specify the data
    X = samples.iloc[:, 0:(samples.shape[1] - 1)]

    # Specify the target labels and flatten the array
    # y= np.ravel(amostras.type)
    y = samples.iloc[:, -1]

    X = X.fillna(0)
    y = y.fillna(0)

    # Apply SelectKBest class to extract top 25 best features
    bestfeatures = SelectKBest(score_func=f_classif, k=25)
    fit = bestfeatures.fit(X, y)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(X.columns)

    #concat two dataframes for better visualization
    feature_scores = pd.concat([df_columns,df_scores],axis=1)
    feature_scores.columns = ['Feature_Name','Score']  # naming the dataframe columns

    print(feature_scores.nlargest(25,'Score'))  # print 25 best features

    feature_scores.nlargest(25, 'Score').to_csv('feature_selected.csv')

    mask = bestfeatures.get_support()
    new_features = X.columns[mask]

    # Creating a new dataframe with the selected features
    X = X[new_features]

    # Split the data up in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test, new_features

# normalize input data

def normalize_data(X_train, X_test):
    # Import `StandardScaler` from `sklearn.preprocessing`
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    # Define the scaler
    # scaler = StandardScaler().fit(X_train)
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)

    # Scale the train set
    X_train = scaler.transform(X_train)

    # Scale the test set
    X_test = scaler.transform(X_test)

    return X_train, X_test

# Reshape data input

def format_3d(df):
    X = np.array(df)
    return np.reshape(X, (X.shape[0], X.shape[1], 1))


def format_2d(df):
    X = np.array(df)
    return np.reshape(X, (X.shape[0], X.shape[1]))

# compile and train learning model

def compile_train(model, X_train, y_train, deep=True):
    if (deep == True):
        import matplotlib.pyplot as plt

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        history = model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=1)
        # model.fit(X_train, y_train,epochs=3)

        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()

        print(model.metrics_names)

    else:
        model.fit(X_train, y_train)  # SVM, LR, GD

    print('Model Compiled and Trained')
    return model


# Testing performance outcomes of the methods

def testes(model, X_test, y_test, y_pred, deep=True):
    if (deep == True):
        score = model.evaluate(X_test, y_test, verbose=1)

        print(score)

    # Alguns testes adicionais
    # y_test = formatar2d(y_test)
    # y_pred = formatar2d(y_pred)

    # Import the modules from `sklearn.metrics`
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, \
        accuracy_score

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print('\nAccuracy')
    print(acc)

    # Precision
    prec = precision_score(y_test, y_pred)  # ,average='macro')
    print('\nPrecision')
    print(prec)

    # Recall
    rec = recall_score(y_test, y_pred)  # ,average='macro')
    print('\nRecall')
    print(rec)

    # F1 score
    f1 = f1_score(y_test, y_pred)  # ,average='macro')
    print('\nF1 Score')
    print(f1)

    # average
    avrg = (acc + prec + rec + f1) / 4
    print('\nAverage (acc, prec, rec, f1)')
    print(avrg)

    return acc, prec, rec, f1, avrg


def test_normal_atk(y_test, y_pred):
    df = pd.DataFrame()
    df['y_test'] = y_test
    df['y_pred'] = y_pred

    normal = len(df.query('y_test == 0'))
    atk = len(y_test) - normal

    wrong = df.query('y_test != y_pred')

    normal_detect_rate = (normal - wrong.groupby('y_test').count().iloc[0][0]) / normal
    atk_detect_rate = (atk - wrong.groupby('y_test').count().iloc[1][0]) / atk

    # print(normal_detect_rate,atk_detect_rate)

    return normal_detect_rate, atk_detect_rate


# Save model and weights

def save_model(model, name):
    from keras.models import model_from_json

    arq_json = 'Models/' + name + '.json'
    model_json = model.to_json()
    with open(arq_json, "w") as json_file:
        json_file.write(model_json)

    arq_h5 = 'Models/' + name + '.h5'
    model.save_weights(arq_h5)
    print('Model Saved')


def load_model(name):
    from keras.models import model_from_json

    arq_json = 'Models/' + name + '.json'
    json_file = open(arq_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    arq_h5 = 'Models/' + name + '.h5'
    loaded_model.load_weights(arq_h5)

    print('Model loaded')

    return loaded_model


def save_Sklearn(model, nome):
    import pickle
    arquivo = 'Models/' + nome + '.pkl'
    with open(arquivo, 'wb') as file:
        pickle.dump(model, file)
    print('Model sklearn saved')


def load_Sklearn(nome):
    import pickle
    arquivo = 'Models/' + nome + '.pkl'
    with open(arquivo, 'rb') as file:
        model = pickle.load(file)
    print('Model sklearn loaded')
    return model


# UPSAMPLE OF NORMAL FLOWS

samples = pd.read_csv('cicddos2019/01-12/export_dataframe_proc.csv', sep=',')

X_train, X_test, y_train, y_test, new_features = train_test(samples)

# junta novamente pra aumentar o numero de normais
X = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes
is_benign = X[' Label'] == 0  # base de dados toda junta

normal = X[is_benign]
ddos = X[~is_benign]

# upsample minority
normal_upsampled = resample(normal,
                            replace=True,  # sample with replacement
                            n_samples=len(ddos),  # match number in majority class
                            random_state=27)  # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([normal_upsampled, ddos])

# Specify the data
X_train = upsampled.iloc[:, 0:(upsampled.shape[1] - 1)]  # DDoS
y_train = upsampled.iloc[:, -1]  # DDoS

input_size = (X_train.shape[1], 1)

del X, normal_upsampled, ddos, upsampled, normal  # , l1, l2

tests = pd.read_csv('cicddos2019/01-12/export_tests_proc.csv', sep=',')

# X_test = np.concatenate((X_test,(tests.iloc[:,0:(tests.shape[1]-1)]).to_numpy())) # testar 33% + dia de testes
# y_test = np.concatenate((y_test,tests.iloc[:,-1]))

del X_test,y_test                            # testar só o dia de testes
X_test = tests.iloc[:,0:(tests.shape[1]-1)]
y_test = tests.iloc[:,-1]
X_test = X_test[new_features]

# print((y_test.shape))
# print((X_test.shape))

X_train, X_test = normalize_data(X_train,X_test)

# Comment next 2 blocks if loading pre-trained models
# Execute them if training new models

model_svm = SVM()
model_lr = LR()
model_knn = kNN()
model_rf = RF()
model_dt = DT()
model_nb = NB()

X_train = np.nan_to_num(X_train)
y_train = y_train.fillna(0)
model_svm = compile_train(model_svm,X_train,y_train,False)
model_lr = compile_train(model_lr,X_train,y_train,False)
model_knn = compile_train(model_knn,X_train,y_train,False)
model_rf = compile_train(model_rf,X_train,y_train,False)
model_dt = compile_train(model_dt,X_train,y_train,False)
model_nb = compile_train(model_nb,X_train,y_train,False)

## Comment next 2 blocks if training new models
## Execute them if loading pre-trained models

# model_svm = load_Sklearn('SVM')
# model_lr = load_Sklearn('LR')
# model_knn = load_Sklearn('kNN-1viz')

results = pd.DataFrame(columns=['Method','Accuracy','Precision','Recall', 'F1_Score', 'Average'])

y_pred = model_svm.predict(X_test)

y_pred = y_pred.round()

acc, prec, rec, f1, avrg = testes(model_svm, X_test, y_test, y_pred, False)

results = results.append({'Method': 'SVM', 'Accuracy': acc, 'Precision': prec, 'F1_Score': f1,
                          'Recall': rec, 'Average': avrg},
                         ignore_index=True)

y_pred = model_lr.predict(X_test)

y_pred = y_pred.round()

acc, prec, rec, f1, avrg = testes(model_lr, X_test, y_test, y_pred, False)

results = results.append({'Method': 'LR', 'Accuracy': acc, 'Precision': prec, 'F1_Score': f1,
                          'Recall': rec, 'Average': avrg},
                         ignore_index=True)

y_pred = model_knn.predict(X_test)

y_pred = y_pred.round()

acc, prec, rec, f1, avrg = testes(model_knn, X_test, y_test, y_pred, False)

results = results.append({'Method': 'kNN', 'Accuracy': acc, 'Precision': prec, 'F1_Score': f1,
                          'Recall': rec, 'Average': avrg},
                         ignore_index=True)

y_pred = model_rf.predict(X_test)

y_pred = y_pred.round()

acc, prec, rec, f1, avrg = testes(model_rf, X_test, y_test, y_pred, False)

results = results.append({'Method': 'RF', 'Accuracy': acc, 'Precision': prec, 'F1_Score': f1,
                          'Recall': rec, 'Average': avrg},
                         ignore_index=True)

y_pred = model_dt.predict(X_test)

y_pred = y_pred.round()

acc, prec, rec, f1, avrg = testes(model_dt, X_test, y_test, y_pred, False)

results = results.append({'Method': 'DT', 'Accuracy': acc, 'Precision': prec, 'F1_Score': f1,
                          'Recall': rec, 'Average': avrg},
                         ignore_index=True)

y_pred = model_nb.predict(X_test)

y_pred = y_pred.round()

acc, prec, rec, f1, avrg = testes(model_nb, X_test, y_test, y_pred, False)

results = results.append({'Method': 'NB', 'Accuracy': acc, 'Precision': prec, 'F1_Score': f1,
                          'Recall': rec, 'Average': avrg},
                         ignore_index=True)

results.to_csv('results.csv', index=False)
print("Finish!")