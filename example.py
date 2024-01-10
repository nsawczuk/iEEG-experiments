import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys 
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from numpy import argmax

def make_model(input_shape,units):

#function that builds the neural network structure. In this case the neural network will have three layers, the number of units/filters in each one will be a parameter

  
    input_layer = keras.layers.Input(input_shape) #The input layer receives a tuple (M,N) where M is the length of the signals and N the number of channels.  

    conv1 = keras.layers.Conv1D(filters=int(units[0]), kernel_size=3, padding="same")(input_layer) #Here's the type of layer, in this case it's a 1D convolutional layer.
    conv1 = keras.layers.BatchNormalization()(conv1) #This is a batch normalization layer
    
    conv1 = keras.layers.ReLU()(conv1) #this is the activation function of the layer

    conv2 = keras.layers.Conv1D(filters=int(units[1]), kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)
 
 #Each layer receives the previous layer as input
    
    conv3 = keras.layers.Conv1D(filters=int(units[2]), kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    x = keras.layers.GlobalAveragePooling1D()(conv3) #This is a global average pooling layer


    output_layer = keras.layers.Dense(1, activation="sigmoid")(x)
    #A dense layer that generates the output. In binary clasification it can have 1 unit and sigmoid as activation function or 2 units and softmax activation function. For multiclass it has K units and softmax activation.
    
    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def method(files):

    data= pd.read_csv(files)
    
    X,y= getXY(data)
    
    
    acc_per_fold = []
    auc_per_fold = []
    loss_per_fold = []
    ppv_per_fold=[]
    npv_per_fold=[]
    sensitivity_per_fold=[]
    f1_score_per_fold=[]
    specificity_per_fold=[]




    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_no = 1



    for train, test in kfold.split(X,y):


        epochs = 150 #maximum of epochs the model will be trained
        batch_size = 128 #size of the batches for training

        optimizer = keras.optimizers.Adam(learning_rate=0.01) #the optimizer handles how the weights of the neural network change during training

        callbacks = [keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=25, min_lr=0.0001    ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, verbose=1),]
#Callbacks for reducing the learning rate and for stopping training if the loss function doesn't improve after X epochs (the patience parameter controls that)


        
        
        X_train= X[train] #Train-test split
        X_test= X[test]
        y_train= y[train]
        y_test=y[test]

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, shuffle=True, random_state=42) #The training set is further split into train-validation
        
        
        scaler = preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        # Standard Scaling applied to the sets
       	
       	
       	units=[64,128,64]	 #number of units per layer for the neural network
        model = make_model((X.shape[1],X.shape[2]),units) #Function that creates the neural network structure, it takes two parameters: the shape of the input (length of the signal, number of channels) and a list with the number of units per layer.
        
        model.compile(  optimizer=optimizer,
            loss= "binary_crossentropy",
                metrics=[
                keras.metrics.BinaryAccuracy(),
                keras.metrics.AUC(),
                keras.metrics.Precision(),
                keras.metrics.Recall(),
            ],) #Here the model is compiled. The loss function is chosen (binary crossentropy in this case, other loss functions can be used such as accuracy). Other metrics are also included, but the model doesn't use them for training, I just included them to see how they evolved during training.


        class_weights = class_weight.compute_class_weight(
                                                  class_weight = "balanced",
                                                  classes = np.unique(y_train),
                                                  y = y_train
                                              )
        class_weights = dict(zip(np.unique(y_train), class_weights))
        #Class weights are set because classes not balanced
        
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
   #evaluation of the model using the testing set

        scores = model.evaluate(X_test, y_test, verbose=0)  #With .evaluate you get the metrics the model was compiled with

        y_prob= model.predict(X_test) #With .predict you get the output of the neural network

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

        ppv_per_fold.append(precision_score(y_test,y_pred))
        npv_per_fold.append(npv)
        sensitivity_per_fold.append(recall_score(y_test,y_pred))
        f1_score_per_fold.append(f1_score(y_test,y_pred))
        specificity_per_fold.append(specif)
        
        
        fold_no= fold_no+1

    f=pd.DataFrame(columns=['subject','acc','auc', 'ppv', 'npv', 'sensitivity',' specificity', 'f1', 'units','method'])
    f.loc[len(f.index)]=[files[6:-6], np.mean(acc_per_fold), np.mean(auc_per_fold),np.mean(ppv_per_fold),np.mean(npv_per_fold), np.mean(sensitivity_per_fold), np.mean(specificity_per_fold), np.mean(f1_score_per_fold),units,'gaussianconv']
    f.to_csv("rescsv/results"+files[6:-6]+"convgaussian.csv")    

method(sys.argv[1])
