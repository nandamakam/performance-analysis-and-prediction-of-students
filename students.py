""" Imported libraries """
import numpy as np
import pandas as pd

""" Reading the data file as DataFrame """
df = pd.read_csv("./data/math.csv", sep=";")    #here pd is taken from pandas library

""" Import Machine Learning helpers """
"""                        sklearn            """
"""Simple and efficient tools for data mining and data analysis
   Accessible to everybody, and reusable in various contexts
   Built on NumPy, SciPy, and matplotlib
   Open source, commercially usable - BSD license         """

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC               # Support Vector Machine Classifier model



""" Train Model and Print Score """
def train_and_score(X, y):
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    clf = Pipeline([
        ('reduce_dim', SelectKBest(chi2, k=2)),
        ('train', LinearSVC(C=100))
    ])
    
    """scores = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=2)
    
    print("Mean Model Accuracy:", np.array(scores).mean())"""
    
    clf.fit(X_train, y_train)

    confuse(y_test, clf.predict(X_test))
    print()
      
      
""" Main Program """
def main():
    print("\n Performance analysis and Prediction of students")
""" Train Model and Print Score """
def train_and_score(X, y):
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    clf = Pipeline([
        ('reduce_dim', SelectKBest(chi2, k=2)),
        ('train', LinearSVC(C=100))
    ])
    
    """scores = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=2)
    
    print("Mean Model Accuracy:", np.array(scores).mean())"""
    
    clf.fit(X_train, y_train)

    confuse(y_test, clf.predict(X_test))
    print()
    # For each feature, encode to categorical values
    class_le = LabelEncoder()
    for column in df[["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]].columns:
        df[column] = class_le.fit_transform(df[column].values)
"""these grades are related with the course subject, Math:
 G1 - first period grade (numeric: from 0 to 20)
 G2 - second period grade (numeric: from 0 to 20)
 G3 - final grade (numeric: from 0 to 20, output target)"""
    # Encode G1, G2, G3 as pass or fail binary values
    for i, row in df.iterrows():
        if row["G1"] >= 10:
            df["G1"][i] = 1
        else:
            df["G1"][i] = 0

        if row["G2"] >= 10:
            df["G2"][i] = 1
        else:
            df["G2"][i] = 0

        if row["G3"] >= 10:
            df["G3"][i] = 1
        else:
            df["G3"][i] = 0

    # Target values are G3. poping of targeted values
    y = df.pop("G3")

    # Feature set is remaining features. it contains remaining G1 and G2 scores
    X = df
    #considering both G1 and G2 Scores
    print("\n\nModel Accuracy Knowing G1 & G2 Scores")
    print("=====================================")
    train_and_score(X, y)                 #train and score both G1 and G2 values

    # Removing grade report 2
    X.drop(["G2"], axis = 1, inplace=True)   #dropping of G2 values
    print("\n\nModel Accuracy Knowing Only G1 Score")
    print("=====================================")
    train_and_score(X, y)                 #train and score G1 values

    # Removing grade report 1
    X.drop(["G1"], axis=1, inplace=True)   #dropping of G1 values
    print("\n\nModel Accuracy Without Knowing Scores")
    print("=====================================")
    train_and_score(X, y)                 #train and score G2 values



main()
