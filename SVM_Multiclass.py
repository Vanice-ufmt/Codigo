# -*- coding: utf-8 -*-
 #!/usr/bin/env python -W ignore::DeprecationWarning
 
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
#from mlxtend.plotting import plot_decision_regions
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from math import *
import time
inicio = time.time()

CLASSE_LABELS = { 'Game':1, 
            'P2P file-sharing':2,
            'HTTP Download':3,
            'Streaming ondemand':4, 
            'Vo IP':5,
            'Remote Session':6,
            'Live Streaming':7,
            'File Transfer':8,
            'P2P Video':9,
            'Web Browsing':10,
            'None':0 }

def getDados(filename):
        dados = []
        with open(filename, 'rb') as spamreader:
                for line in spamreader:
                        aux = np.asarray(line.replace('[','').replace(']','').split(','), dtype=float)
                        dados.append(aux)
        return dados
 
def getTarget(filename): 
        target = []        
        with open(filename, 'rb') as spamreader:
                for line in spamreader:
                        target.append(line.replace('\n',''))
        return target

print "Lendo base de treino..."
dados = getDados('DATABASE_SVM_INDIVIDUAL.txt')
target = getTarget('DATABASE_SVM_INDIVIDUAL_TARGET.txt')


#Converte listas de dados e targets em numpy arrays
dados = np.array(dados)
target = np.array(target)

#Variáveis de treino (Individuais), onde x � a base de dados e o y � a classe.

X = dados
y = target

print '-------------------'


# title for the plots
titles = (#'SVC with linear kernel'
         #'SVC with sigmoid kernel'
         #'SVC with RBF kernel'
         'SVC with polynomial (degree=3) kernel'
          )


# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
#C = 1.0  # SVM regularization parameter

models = (#svm.SVC(kernel='linear', C=50),
          #svm.SVC(kernel='sigmoid', C=50),
          #svm.SVC(kernel='rbf', C=50),
          svm.SVC(kernel='poly', C=50),
          )
models = (clf.fit(X, y) for clf in models)

print "Lendo base de teste..."
dados_teste = getDados('DATABASE_SVM_COLETIVO.txt')
target_teste = getTarget('DATABASE_SVM_COLETIVO_TARGET.txt')


print '-------------------'

#Converte listas de dados e targets em numpy arrays
dados_teste = np.array(dados_teste)
target_teste = np.array(target_teste)

#Variáveis de teste (Coletivos)
X_test = dados_teste
y_test = target_teste

b = 0


for i in models:	
	print "Classifying model:", titles[b]
        pred = i.predict(X_test)
        cr = classification_report(y_test,pred)
        print cr
        cm = confusion_matrix(y_test, pred)
        #Normalizando diagonais para obter acurácia
        cm2 = cm
        cm2 = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]
        acc = 'Accuracy:'+ str(np.nan_to_num(cm2.diagonal()))+ '\n'
        print acc
        f = open('Matriz de confusão - '+titles[b]+'.txt', 'w')
        f.write('Classification Report \n')
        f.write(cr)
        f.write('\n')
        f.write(acc)
        f.write('\n')
        f.write('Confusion Matrix: \n')
        f.write(str(cm))
        f.close()
	b +=1	



print '\nTempo de execução:', time.time() - inicio, 'segundos.'
