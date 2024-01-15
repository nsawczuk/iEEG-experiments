import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from numpy import argmax
#Gaussian function from the oscillation
def gaussian(x,a,p,w):
    return a*np.exp(-((x-p)**2)/(2*w**2))

def get_timeSeries_trial(Data,k):
    y=np.zeros(k)
    x=np.linspace(0,3,k)

    pkMag= np.array(Data['pkMag'])
    tStart= np.array(Data['tStart'])
    tEnd= np.array(Data['tEnd'])
    tDur= np.array(Data['tDur'])
    for i in range(len(pkMag)):
        y=y+gaussian(x,pkMag[i], tStart[i]+(tEnd[i]-tStart[i])/2, tDur[i])
    return y

def get_timeSeries(Data, electrodes):
    nTrials=len(Data['trial'].unique())
    if Data['tEnd'].max()<1.5:

        timeSeries= np.zeros((nTrials,150,electrodes.shape[0]))
        k=150
    else:
        timeSeries= np.zeros((nTrials,3000,electrodes.shape[0]))
        k=3000
    for i in range(electrodes.shape[0]):
        auxData=Data[Data['Electrode name']==electrodes[i]]
        for j in range(nTrials):
            timeSeries[j,:,i]= get_timeSeries_trial(auxData[auxData['trial']==j],k)
    return timeSeries
    
def make_model(input_shape,units):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=int(units[0]), kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=int(units[1]), kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=int(units[2]), kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    x = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def gaussian_method(highgamma_recall,threshold, splits):

    highGamma_recall= pd.read_csv(highgamma_recall)
    bet= 'files_beta/'+highgamma_recall[6:-6]+'b.csv'
    beta_recall= pd.read_csv(bet)
    
    archivo='file1/'+highgamma_recall[6:-19]+'lBA4hg.csv'
	
    

#    y=np.array(highGamma.groupby('trial')['Recalled'].mean())

    acc_per_fold = []
    auc_per_fold = []
    loss_per_fold = []

    ppv_per_fold=[]
    npv_per_fold=[]
    sensitivity_per_fold=[]
    f1_score_per_fold=[]
    specificity_per_fold=[]
    units=[64,128,64]



    fold_no = 1

	

    recalls= np.array(pd.read_csv(archivo).groupby('trial')['Recalled'].mean())
    electrodes= pd.read_csv(archivo)['Electrode name'].unique()
    f=pd.DataFrame(columns=['subject','acc','auc', 'ppv', 'npv', 'sensitivity',' specificity', 'f1', 'units','method','positives','threshold'])    
    y=[]
    recalls_per_session=[]
    timeSeriesB=get_timeSeries(beta_recall,electrodes)
    timeSeriesHG= get_timeSeries(highGamma_recall,electrodes)
    X= np.concatenate([timeSeriesHG,timeSeriesB],axis=2)    
    for i in range(highGamma_recall.trial.nunique()):
    	recalls_per_session.append(recalls[i*12:i*12+12].sum())

#    threshold=np.array(recalls_per_session).mean()
            	
    for i in range(len(recalls_per_session)):
        if recalls_per_session[i]<1:
        	y.append(0)
        else:
        	y.append(1)

    y=np.array(y)

    rkf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=5, random_state=2652124)
    for i, (train, test) in enumerate(rkf.split(X,y)):    
        epochs = 70
        batch_size = 32
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        callbacks = [keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=25, min_lr=0.0001    ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, verbose=1),]
        

    
        X_train= X[train]
        X_test= X[test]
        y_train= y[train]
        y_test=y[test]

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, shuffle=True , random_state=42)
        scaler = preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        model = make_model((X.shape[1],X.shape[2]),units)
        model.compile(  optimizer=optimizer,
            loss= "binary_crossentropy",
                metrics=[
                keras.metrics.BinaryAccuracy(),
                keras.metrics.AUC(),
                keras.metrics.Precision(),
                keras.metrics.Recall(),
            ],)


        class_weights = class_weight.compute_class_weight(
                                                  class_weight = "balanced",
                                                  classes = np.unique(y_train),
                                                  y = y_train
                                              )
        class_weights = dict(zip(np.unique(y_train), class_weights))
        y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
        y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
        y_val= np.asarray(y_val).astype('float32').reshape((-1,1))

        print(f'Fold {fold_no} ...')

        history = model.fit(
              X_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks,
              validation_data=(X_val,y_val),
              verbose=1,
              class_weight=class_weights,
          )

        scores = model.evaluate(X_test, y_test, verbose=0)

        y_prob= model.predict(X_test)

        auc_per_fold.append(roc_auc_score(y_test,y_prob))


        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        J = tpr - fpr
        ix = argmax(J)
        best_thresh = thresholds[ix]

        y_pred= (y_prob>best_thresh)

        tn,fp,fn,tp= confusion_matrix(y_test,y_pred).ravel()
        print(confusion_matrix(y_test,y_pred))
        npv= tn/(tn+fn)
        specif= tn/(tn+fp)

        acc_per_fold.append(accuracy_score(y_test, y_pred))

        loss_per_fold.append(scores[0])

        ppv_per_fold.append(precision_score(y_test,y_pred,zero_division=0))
        npv_per_fold.append(npv)
        sensitivity_per_fold.append(recall_score(y_test,y_pred))
        f1_score_per_fold.append(f1_score(y_test,y_pred))
        specificity_per_fold.append(specif)
        fold_no= fold_no+1
        f.loc[len(f.index)]=[highgamma_recall[6:-6], np.mean(acc_per_fold), np.mean(auc_per_fold),np.mean(ppv_per_fold),np.mean(npv_per_fold), np.mean(sensitivity_per_fold), np.mean(specificity_per_fold), np.mean(f1_score_per_fold),units,'gaussianconv', y.sum(), threshold]
    f.to_csv("exp3results/results"+highgamma_recall[6:-6]+"_exp3_convgaussian.csv")    

gaussian_method(sys.argv[1],sys.argv[2],sys.argv[3])
