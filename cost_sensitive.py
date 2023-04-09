from sklearn.preprocessing import OneHotEncoder, StandardScaler
from collections import Counter
from sklearn.datasets import  fetch_openml
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter


def transformData(X):
    one_hot_encoder = make_column_transformer(
        (OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
        make_column_selector(dtype_include='category')),
        remainder='passthrough')

    X = one_hot_encoder.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X

def costMinimization(X_train, X_test, y_train, y_test, clf, cost_m):
    print("\ncost minimization without probability calibration")
    # Encode labels to match train and test data
    model = clf.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)
    y_pred = np.argmin(np.matmul(y_pred_prob, np.array(cost_m).T), axis=1)
    print(classification_report(y_test, y_pred))
    conf_m = confusion_matrix(y_test, y_pred).T # transpose to align with slides
    print(conf_m) 
    print(np.sum(conf_m * cost_m))

def costBasedSampling(X_train, X_test, y_train, y_test, clf, cost_m):
        print("with undersampling")
        #change the examples of class 1 to see what happens
        sampler = RandomUnderSampler(sampling_strategy={0: 149, 1: 100}, random_state=1) 
        X_rs, y_rs = sampler.fit_resample(X_train, y_train)
        print(Counter(y_rs))

        model = clf.fit(X_rs, y_rs)
        y_pred = clf.predict(X_test)

        print(classification_report(y_test, y_pred))
        conf_m = confusion_matrix(y_test, y_pred).T # transpose to align with slides
        print(conf_m)
        loss = np.sum(conf_m * cost_m)
        print("%d\n" %loss)

        print("with oversampling")
        sampler = RandomOverSampler(sampling_strategy={0: 350, 1: 750}, random_state=1) 
        X_rs, y_rs = sampler.fit_resample(X_train, y_train)
        print(Counter(y_rs))

        model = clf.fit(X_rs, y_rs)
        y_pred = clf.predict(X_test)

        print(classification_report(y_test, y_pred))
        conf_m = confusion_matrix(y_test, y_pred).T # transpose to align with slides
        print(conf_m)
        loss = np.sum(conf_m * cost_m)
        print("%d\n" %loss)

        print("with combination")
        sampler = RandomUnderSampler(sampling_strategy={0: 149, 1: 150}, random_state=1)
        X_rs, y_rs = sampler.fit_resample(X_train, y_train)
        sampler = RandomOverSampler(sampling_strategy={0: 250, 1: 150}, random_state=1)
        X_rs, y_rs = sampler.fit_resample(X_rs, y_rs)
        print(Counter(y_rs))

        model = clf.fit(X_rs, y_rs)
        y_pred = clf.predict(X_test)

        print(classification_report(y_test, y_pred))
        conf_m = confusion_matrix(y_test, y_pred).T # transpose to align with slides
        print(conf_m)
        loss = np.sum(conf_m * cost_m)
        print("%d\n" %loss)

def costBasedWeighting(X_train, X_test, y_train, y_test, clf, cost_m):
    print("\nwith weights")
    # now create the sample weights according to y
    weights = np.zeros(y_train.shape[0])
    weights[np.where(y_train == 1)] = 1;
    weights[np.where(y_train == 0)] = 4;
    model = clf.fit(X_train, y_train, weights)
    pred_test = clf.predict(X_test)
    print(classification_report(y_test, pred_test))
    conf_m = confusion_matrix(y_test, pred_test).T # transpose to align with slides
    print(conf_m)
    loss = np.sum(conf_m * cost_m)
    print("%d\n" %loss)

    print("\nwith weights (alternative)")
    clf.class_weight = {0: 4, 1: 1}
    model = clf.fit(X_train, y_train)
    pred_test = model.predict(X_test)
    print(classification_report(y_test, pred_test))
    conf_m = confusion_matrix(y_test, pred_test).T # transpose to align with slides
    print(conf_m)
    loss = np.sum(conf_m * cost_m)
    print("%d\n" %loss)

if __name__ == '__main__':
    # Retrieve german credit dataset from openml
    X, y = fetch_openml("credit-g", version=1, as_frame=True, parser='auto', return_X_y=True)

    data = transformData(X)

    # Get the class distribution for the target variable y
    class_distribution = y.value_counts().sort_index()

    print(class_distribution)

    # Define cost matrix
    cost_m = [[0, 1], 
            [5, 0]]

    # Split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=0, stratify=y)

    names = ['random forest', 'linear SVM', 'Naive Bayes']
    classifiers = [RandomForestClassifier(n_estimators=100, random_state=0), 
                SVC(kernel='linear',  probability=True),
                GaussianNB()]

    for name, clf in zip(names, classifiers):
        print(name)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))
        conf_m = confusion_matrix(y_test, y_pred).T # transpose to align with slides
        print(conf_m) 
        print(np.sum(conf_m * cost_m))

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        
        # Minimizing expected cost
        costMinimization(X_train, X_test, y_train, y_test, clf, cost_m)

        # Cost-based sampling
        costBasedSampling(X_train, X_test, y_train, y_test, clf, cost_m)

        # Cost-based weighting
        costBasedWeighting(X_train, X_test, y_train, y_test, clf, cost_m)