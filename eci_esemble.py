# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 16:24:30 2016

Analysis of online users data using ensemble methods

@author: Óscar Barquero Pérez
         Rebeca Goya Esteban
         Carlos Figuera Pozuelo
"""

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import sklearn.preprocessing as prepro
from sklearn.cross_validation import train_test_split
import matplotlib.pylab as plt
from sklearn.grid_search import GridSearchCV
import sklearn.metrics as metrics
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.decomposition import PCA


def read_data(filename):
    """
    Function that performs an initial exploratory data analysis using pandas. 
    This allows to verify the collinearity problem.
    
    inputs -   fname: filename with the data in csv format
    outputs -  X,y : X explicative variables matrix and y response variable in 
                    numpy format
    """
    
    # At this point we are going to think that we have a curated and balanced dataset
    # from R.
    
    eci_data = pd.read_csv(filename,delimiter=" ") # this reads the data using panda
    #remove the first column which is only an id
    #
    print eci_data.describe()
    #TODO may be there is need to perform some actions on the data, some NAN conversion 
    #and some categorization on the data
    

    #According to my code in R we have to look at the following variables to be sure they are 
    #correctly preserved

    # sid_avgduration: duración media de las sesiones (¿qué significa NaN?), -> asumimos que cuando NaN, esto significa 0 en los últimos 30 días
    # count_trans: num de transacciones totales por navegador (¿qué significa NaN?) -> asumimos que cuando NaN, ésto significa 0 en los últimos 30 días
    # bounces: número de veces que rebota con la página (¿qué es NaN?) -> cuanod NaN asumimos 0
    # pageviews: total de páginas vistas del usuario (¿qué es NaN?) -> cuando es NaN asumimos 0
    # pageviews_persid: media de páginas vista por sesión del usuario (¿qué es NaN?) -> cuando es NaN asumimos 0
    # timeOnSite: tiempo en el site (¿qué es NaN?) -> asumismos que NaN quiere decir que en los 30 días no ha visitado site ECI.
    # totalTransactionRevenue: ingresos del usuario -> sería una y2 para el caso de multclasificación o regresión.

    #Reacondicionamos el dataframe de acuerdo a las consideraciones anteriores
    # el resultado es asignar a todos los na dentro del dataframe el valor 0
    #eci[is.na(eci)] = 0

    #We change every NaN in the dataframe for 0.
    eci_data.fillna(0, inplace = True)  

    #2) Convert data into numpy matrix and extract the names of each column

    col_names = list(eci_data.columns.values)
    
    #Obtain the name of the numerical variables which are going to be normalized
    num_variables = list(eci_data.select_dtypes(exclude=['bool']).columns.values)
    
    X_aux = eci_data.as_matrix()
    
    #3) Convert into an X and y 
    y_idx = col_names.index('transaction')
    y = X_aux[:,y_idx:y_idx+1]
    X = X_aux[:,np.arange(len(col_names))!= y_idx]
    X = X.astype(float)
    y = y.astype(int)
    
    
    return eci_data, col_names, num_variables, X, y


def split_train_test_PCA(X,y,col_names,numeric_var,plot = False,norm = False,pca=False):
    """
    In this function we are going to : (1)balance the data, 
    since we have a lot of data we are goin to downsample the
    greater class. However other approaches are advised. (2) split into
    training(indeed validation) and test sets. (3) perform PCA
    """
    
    #get indexes with y == 1 and y == 0, and then 
    #get the same number of values for 0 and 1, extract also the balanced indexes from X
    # matrix
    # First shuffle data and then take the first len(np.where(y==1)) data from X and y 
    #that fulfiil y == 0

    X, y = shuffle(X,y)
    
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    num_class1 =  len(idx_1)
    
    #X_balanced
    idx_balanced = np.concatenate((idx_0[:num_class1],idx_1))
    
    y_balan = y[idx_balanced]
    X_balan = X[idx_balanced,:]
    
    X_balan, y_balan = shuffle(X_balan, y_balan)
    
    #split between train and test
    X_train, X_test, y_train, y_test = train_test_split(X_balan, y_balan, test_size=0.15,random_state=42)
    
    if norm == True:
        #then normalize the data. We need the information about which variables are numeric
        #categorical variables are not normalized
        idx_to_norm = []
        for num_var in numeric_var:
            if num_var != 'transaction':
                idx_to_norm.append(col_names.index(num_var))
  
        X_balan_norm =  X_train[:,idx_to_norm]
        
        #normalization
        scaler = prepro.StandardScaler()
        scaler.fit(X_balan_norm)
        X_balan_norm = scaler.transform(X_balan_norm)
        X_train[:,idx_to_norm] = X_balan_norm
    #normalization end
    
    if pca == True:
        pca_object = PCA()
        pca_object.fit(X_train)
        #let's use all the components
        X_train_proj = pca_object.transform(X_train)
        #Then use pca_object to obtain the projections for X_test
    else:
        pca_object = []
        X_train_proj = X_train
    #Some plots    
    if plot == 1:
        
        for i in range(4):
            plt.figure()
            plt.bar(np.arange(np.shape(X_train)[1]), pca_object.components_[i])
            if i == 0:
                plt.ylabel('1st component')
            elif i == 1:
                plt.ylabel('2nd component')
            else:
                plt.ylabel('3rd component')
            axis_c = plt.gca()
            axis_c.set_xticklabels(col_names[:-1],fontsize = 7)
            axis_c.set_xticks(axis_c.get_xticks() + 0.5)
        plt.figure()
        plt.plot(np.cumsum(pca_object.explained_variance_ratio_),'.-')
        plt.xlabel('Number of components')
        plt.ylabel('Accumulated explained variance')
        plt.figure()
        plt.semilogy(np.cumsum(pca_object.explained_variance_ratio_),'.-')
        plt.xlabel('Number of components')
        plt.ylabel('Accumulated explained variance')
    preprocessed_data = {'X_train':X_train,'y_train':y_train,'X_test':X_test,
    'y_test':y_test,'X_train_proj':X_train_proj,'Scaler':scaler,
    'idx_to_norm':idx_to_norm,'pca_object':pca_object}
    
    return preprocessed_data




def training_models(preprocessed_data, pca = False):
    """
    Functions that performs the training of the different models
    inputs-   preprocessed_data: dictionary with all the data
    outputs-  dictionaries with the trained models
    """
    X_train = preprocessed_data['X_train_proj']
    y_train = preprocessed_data['y_train']
    train_size = len(y_train)
    n_features = X_train.shape[1]

    #Tuning parameters of SVM
    cv_folds = 5
    tuning_parameters_svm = {'kernel': ['rbf'], 'gamma': np.logspace(-4, 2,30),'C': np.logspace(-2, 2,30)}
    #tuning_parameters_svm = {'kernel': ['rbf'], 'gamma': np.logspace(-4, 2,10),'C': np.logspace(-2, 2,5)}
    param_grid_brt = {'n_estimators':[4000],'learning_rate': np.logspace(-4,0,30),'max_depth': [2,3,4,10],'min_samples_leaf': [3,5,10]}
    #param_grid_brt = {'n_estimators':np.arange(2000,5000,2),'learning_rate': np.logspace(-4,0,2),'max_depth': [2,3,4],'min_samples_leaf': [3,5]}
    #evaluation metrics
    scorer = metrics.make_scorer(metrics.mean_squared_error,greater_is_better = False)
    
    
    #Real training
    

    
#    
#    #svm + pca
    print('------------------------')
    print('SVM + pca')
    print (' ')
    svm = GridSearchCV(SVC(C=1), param_grid = tuning_parameters_svm, cv=5, scoring = scorer, n_jobs = -1,verbose = 1)
    svm.fit(X_train,np.ravel(y_train))
    #svm.best_estimator_

    #boosted regression tress + pca
    print('------------------------')
    print('BRT + pca')
    print (' ')
    brt = GridSearchCV(GradientBoostingClassifier(), cv = 5,param_grid = param_grid_brt,scoring = scorer,n_jobs = -1,verbose = 1)
    brt.fit(X_train,np.ravel(y_train))
    #regress_models['brt_pca']['brt_pca'] = brt.best_estimator_    
    
    #saving models
    #TO_DO 
    #save regress models 
    if pca == True:
        joblib.dump(svm,'./svm.ojki',compress = 1,cache_size = 1e7)
        joblib.dump(brt,'./brt.ojki',compress = 1,cache_size = 1e7)
        joblib.dump(preprocessed_data,'./prepro_data.ojki',compress = 1, cache_size = 1e7)
    else:
        joblib.dump(svm,'./svm_no_pca.ojki',compress = 1,cache_size = 1e7)
        joblib.dump(brt,'./brt_no_pca.ojki',compress = 1,cache_size = 1e7)
        joblib.dump(preprocessed_data,'./prepro_data_no_pca.ojki',compress = 1, cache_size = 1e7)
    #joblib.dump(pca_pls,'./paper_svm_brt/python_analysis/pca_pls.ojki',compress = 1,cache_size = 1e7)
    #joblib.dump(train_GSCV_scores,'./paper_svm_brt/python_analysis/train_GSCV_scores.ojki',compress = 1,cache_size = 1e7)
    return svm,brt
        
def test_accuracy(svm,brt,prepro_data,col_names,pca = False):
    #in this function we are really computing the accuracy. As well as representing
    #the solution of the best brt algorithm
    #get X_test and y_test
    X_test = prepro_data['X_test']
    y_test = prepro_data['y_test']
    #get scaling factors and apply them to X_test, note only to those index that are
    #numeric, verify this with the previous normalization in prepro fucntion
    scaler = prepro_data['Scaler']
    pca_ob = prepro_data['pca_object']
    idx_to_norm = prepro_data['idx_to_norm']
    
    #get data to normalize
    X_test_norm = X_test[:]
    X_to_norm = X_test[:,idx_to_norm]
    X_norm = scaler.transform(X_to_norm)
    X_test_norm[:,idx_to_norm] = X_norm
    
    #get pca object and project X_test
    if pca == False:
        X_test_proj = X_test_norm
    else:
        X_test_proj = pca_ob.transform(X_test_norm)
    

    #get the best estimator
    svm_best = svm.best_estimator_
    brt_best = brt.best_estimator_
    
    print svm_best
    print brt_best
    
    y_pred_svm = svm_best.predict(X_test_proj)
    y_pred_brt = brt_best.predict(X_test_proj)
    
    print np.mean(y_test == y_pred_svm)
    acc_svm = metrics.accuracy_score(np.ravel(y_test),y_pred_svm)
    acc_brt = metrics.accuracy_score(np.ravel(y_test),y_pred_brt)  
    
    print "Accuracy svm: ", acc_svm
    print "Accuracy brt: ", acc_brt

    print "-----------------"
    print " Classification report SVM"
    print(metrics.classification_report(y_test, y_pred_svm))   

    print "-----------------"
    print " Classification report BRT"
    print(metrics.classification_report(y_test, y_pred_brt)) 

    feature_importance = brt_best.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
   # plt.subplot(1, 2, 2)
    plt.barh(pos,feature_importance[sorted_idx], align='center')
    cl_n = np.array(col_names)
    plt.yticks(pos, cl_n[sorted_idx])   
    plt.xlabel('Realtive importance')
    
    #plot some partial dependence plots    
    plt.figure()
    sorted_idx_list = sorted_idx.tolist()
    features = sorted_idx_list[0:4] #+ [(sorted_idx_list[0],sorted_idx_list[1])]
    
    fig, axs = plot_partial_dependence(brt_best, prepro_data['X_train_proj'],features,n_jobs = -1, grid_resolution = 2000)
    fig.suptitle('Partial dependence of ECI data')
    plt.subplots_adjust(top = 0.9)
    
    #boxplot for predicted probabilities
    
    pred_prob = brt_best.predict_proba(X_test_proj)
    y_1 = pred_prob[y_test.ravel() == 1,1]
    y_0 = pred_prob[y_test.ravel() == 0,1,]
    
    plt.figure()
    plt.boxplot([y_0.tolist(),y_1.tolist()],labels = ['0','1'])
    plt.ylabel('Predicted probability')
    plt.xlabel('Transaction')
    
    plt.figure()
    
    xx = np.arange(len(pred_prob))
    
    print len(xx)
    print len(pred_prob)
    print len(y_test)
    plt.scatter(xx[y_test.ravel() == 0],pred_prob[y_test.ravel() == 0,1],s = 60,c = 'r',alpha= 0.6,label = 'Trans = 0')
    plt.scatter(xx[y_test.ravel() == 1],pred_prob[y_test.ravel() == 1,1],s = 60,c = 'b',alpha= 0.6,label = 'Trans = 1')
    plt.legend(scatterpoints = 1)
    plt.ylabel('Predicted probability')
    plt.xlabel('# sample')
    
    
def main():
    
    """
    Main function:
    """
    
    #1) read the data and verify that all the data is in the good fashion
    
    eci_data, col_names,num_variables, X, y = read_data('curated_balanced_eci.csv')
    
    pca = False
    prepro_data = []
    
    ip = raw_input('You wanna (1)test / (2) train: [1]')
    if ip == '2':
        prepro_data = split_train_test_PCA(X,y,col_names,num_variables,plot = False,norm = True,pca= pca)
        brt_model, svm_model = training_models(prepro_data,pca = pca)
        #load svm and brt models
    else:
        if pca == False:
            svm = joblib.load('svm_no_pca.ojki')
            brt = joblib.load('brt_no_pca.ojki')
            prepro_data_test = joblib.load('prepro_data_no_pca.ojki')
        else:
            svm = joblib.load('svm.ojki')
            brt = joblib.load('brt.ojki')
            prepro_data_test = joblib.load('prepro_data.ojki')
        test_accuracy(svm,brt,prepro_data_test,col_names,pca)
    
    
    return eci_data, col_names, X, y, prepro_data
    
if __name__ == '__main__':
   eci_data, col_names, X, y, prepro_data,  =  main()
