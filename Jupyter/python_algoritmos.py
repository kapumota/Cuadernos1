#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
algoritmos_ml.py

Este script contiene ejemplos de los principales algoritmos de machine learning:
1. Regresión lineal
2. Regresión logística
3. Árbol de decisión
4. k-vecinos (k-NN)
5. Máquina de vectores de soporte (SVM)

Se utilizan los conjuntos de datos 'diabetes' (para regresión) e 'iris' (para clasificación) de scikit-learn.
"""

from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import r2_score, accuracy_score

def linear_regression_example():
    """Ejemplo de Regresión Lineal usando el conjunto de datos diabetes."""
    # Cargar el conjunto de datos de diabetes
    data = load_diabetes()
    X, y = data.data, data.target

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo de regresión lineal
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Predecir y evaluar el modelo
    y_pred = lr.predict(X_test)
    score = r2_score(y_test, y_pred)
    print("Regresión Lineal (Diabetes) - R^2:", score)

def logistic_regression_example():
    """Ejemplo de Regresión Logística usando el conjunto de datos iris."""
    # Cargar el conjunto de datos iris
    data = load_iris()
    X, y = data.data, data.target

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo de regresión logística
    logreg = LogisticRegression(max_iter=200)
    logreg.fit(X_train, y_train)

    # Predecir y evaluar el modelo
    y_pred = logreg.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Regresión Logística (Iris) - Accuracy:", acc)

def decision_tree_example():
    """Ejemplo de Árbol de Decisión usando el conjunto de datos iris."""
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Árbol de Decisión (Iris) - Accuracy:", acc)

def knn_example():
    """Ejemplo de k-vecinos usando el conjunto de datos iris."""
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("k-Vecinos (Iris) - Accuracy:", acc)

def svm_example():
    """Ejemplo de máquina de vectores de soporte (SVM) usando el conjunto de datos iris."""
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("SVM (Iris) - Accuracy:", acc)

if __name__ == "__main__":
    print("Ejemplo de regresión lineal (Diabetes dataset):")
    linear_regression_example()
    print("\nEjemplo de regresión logística (Iris dataset):")
    logistic_regression_example()
    print("\nEjemplo de árbol de decisión (Iris dataset):")
    decision_tree_example()
    print("\nEjemplo de k-vecinos (Iris dataset):")
    knn_example()
    print("\nEjemplo de máquina de vectores de soporte (Iris dataset):")
    svm_example()

