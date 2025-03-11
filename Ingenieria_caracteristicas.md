## Ingeniería de características

La **ingeniería de características** es el proceso de transformar datos brutos en variables (características) útiles que permitan a los algoritmos de machine learning aprender patrones relevantes para la tarea a resolver. Este proceso no solo consiste en la limpieza de datos, sino también en la selección, transformación, combinación y creación de nuevas variables que capturen la esencia del problema. La calidad de estas características puede determinar en gran medida el éxito del modelo, ya que características bien diseñadas pueden simplificar la tarea del algoritmo, mientras que variables inadecuadas o mal procesadas pueden llevar a modelos ineficientes o sobreajustados.

### Tipos de características

#### 1. Características continuas

- **Definición:** Son variables numéricas que pueden tomar un rango amplio o incluso infinito de valores dentro de un intervalo.  
- **Ejemplos:** Temperatura, tiempo, distancias o brillo de píxeles en imágenes.  
- **Consideraciones:**  
  - **Escalado y normalización:** Dado que algunas técnicas de machine learning (como KNN o regresiones lineales) son sensibles a la escala de los datos, es importante aplicar técnicas como el Min-Max Scaling o la normalización Z-Score.
  - **Transformaciones:** En casos de distribuciones asimétricas, se puede aplicar una transformación logarítmica para aproximar una distribución normal.

#### 2. Características categóricas

- **Definición:** Variables que representan categorías o grupos discretos.  
- **Ejemplos:** Género, tipo de trabajo, país de residencia o nivel educativo.  
- **Consideraciones:**  
  - **Codificación:** La mayoría de los algoritmos requieren datos numéricos, por lo que se deben convertir mediante técnicas como one-hot encoding, label encoding, binary encoding o target encoding.
  - **Cardinalidad:** Variables con muchos valores únicos pueden ocasionar un aumento excesivo en la dimensionalidad, por lo que se deben aplicar técnicas que controlen este efecto.
  - **Variables ordinales:** Aunque son categóricas, poseen un orden inherente (por ejemplo, niveles de satisfacción) y pueden ser codificadas de manera que se preserve esta jerarquía.

#### 3. Otras variables

- **Variables Ordinales:** Representan categorías con un orden natural, aunque su diferencia numérica no sea constante.  
- **Variables Booleanas:** Pueden representarse como 0 y 1, siendo útiles para indicar presencia o ausencia de una característica.

### Técnicas de ingeniería de características

La transformación de los datos en un formato adecuado para el modelado implica diversas técnicas:

#### a) Codificación de variables categóricas

- **One-hot encoding:**  
  Se crea una nueva columna binaria por cada categoría de la variable original. Por ejemplo, la variable `workclass` con cuatro categorías se transforma en cuatro columnas, cada una con valor 1 o 0 según la presencia de la categoría.  
- **Label encoding:**  
  Se asigna un entero a cada categoría. Es útil cuando existe un orden natural, pero puede inducir relaciones erróneas si las categorías no tienen una secuencia lógica.  
- **Target (Mean) encoding:**  
  Reemplaza cada categoría por el valor medio de la variable objetivo, aunque se debe tener cuidado para evitar la fuga de información mediante técnicas de validación cruzada.  
- **Binary encoding:**  
  Convierte las categorías a una representación binaria, lo que puede reducir la dimensionalidad frente a One-Hot Encoding en variables con alta cardinalidad.  
- **Frequency encoding:**  
  Reemplaza cada categoría por su frecuencia de aparición en los datos, lo que puede capturar la importancia de la prevalencia en ciertas aplicaciones.

#### b) Escalado y normalización

- **Min-Max scaling:**  
  Transforma los valores numéricos a un rango específico, comúnmente entre 0 y 1.  
- **Z-Score normalization (estandarización):**  
  Reescala los datos para que tengan una media 0 y una desviación estándar 1.  
- **Log transformation:**  
  Es útil para variables con distribuciones sesgadas, permitiendo una distribución más simétrica.  
- **Robust scaling:**  
  Utiliza la mediana y los cuartiles, siendo menos sensible a valores atípicos.

#### c) Generación de nuevas características

La creación de nuevas variables puede revelar interacciones y relaciones no lineales entre los datos:

- **Interacciones y combinaciones:**  
  Se pueden crear nuevas variables a partir del producto, cociente o sumas de dos o más características.  
- **Transformaciones no lineales:**  
  Aplicar funciones como la cuadrática, raíz cuadrada o logaritmo para capturar relaciones complejas.  
- **Características temporales:**  
  A partir de datos de fecha/hora se pueden extraer variables como día, mes, año, hora, estacionalidades y tendencias.  
- **Reducción de dimensionalidad:**  
  Técnicas como PCA, t-SNE o UMAP permiten condensar la información en un número menor de variables que retienen la mayor parte de la variabilidad.  
- **Autoencoders:**  
  Redes neuronales que aprenden representaciones comprimidas de los datos, útiles tanto para reducción de ruido como para detección de anomalías.
- **Discretización:**  
  Convertir variables continuas en categorías (bins) puede simplificar la modelación en ciertos casos, especialmente cuando se espera que la relación entre la variable y el objetivo sea no lineal.

#### d) Manejo de datos faltantes

El tratamiento adecuado de datos incompletos es fundamental para evitar sesgos y mejorar el desempeño del modelo:

- **Eliminación de registros:**  
  Se remueven filas con pocos datos faltantes, siempre y cuando no se pierda información significativa.  
- **Imputación de valores:**  
  Se reemplazan los valores faltantes utilizando la media, mediana, moda o métodos basados en modelos (como KNN o regresión).  
- **Creación de indicadores:**  
  La ausencia de datos puede ser informativa por sí sola, por lo que se pueden crear variables binarias que indiquen si un dato estaba ausente.

#### e) Selección de características

No todas las variables contribuyen de igual manera al modelo. La selección de las características más relevantes ayuda a reducir la complejidad y evita el sobreajuste:

- **Métodos basados en filtros:**  
  Uso de métricas estadísticas (por ejemplo, correlación, pruebas Chi-cuadrado) para seleccionar variables con alta relevancia.  
- **Métodos basados en wrappers:**  
  Técnicas iterativas como Recursive Feature Elimination (RFE) que eliminan variables de manera secuencial evaluando el rendimiento del modelo.  
- **Métodos embebidos (embedded):**  
  Algoritmos como LASSO o modelos basados en árboles que incorporan la selección de características durante el proceso de entrenamiento.

#### f) Técnicas avanzadas y automatización

- **Ingeniería automática de características:**  
  Herramientas como Featuretools automatizan la generación de nuevas variables a partir de datos relacionales y series temporales.  
- **Clusterización para segmentación:**  
  Aplicar algoritmos de clustering para agrupar datos similares y luego usar estos grupos como una nueva característica en el modelo.


### Algoritmos y herramientas en ingeniería de características

Diversos algoritmos y herramientas se utilizan para identificar y transformar las características relevantes:

- **Árboles de decisión y ensambles:**  
  Modelos como Random Forests y Gradient Boosting no solo son potentes en la predicción, sino que también proporcionan medidas de importancia para cada variable, ayudando a identificar cuáles son las características más influyentes.
- **Análisis de componentes principales (PCA):**  
  Esta técnica de reducción de dimensionalidad permite eliminar redundancias y capturar la variabilidad en un conjunto reducido de variables.
- **Autoencoders:**  
  Utilizados en deep learning, aprenden una representación comprimida de los datos que puede servir para la reducción de ruido y detección de patrones ocultos.
- **Embeddings y representaciones:**  
  En el procesamiento de lenguaje natural, técnicas como Word2Vec, FastText y TF-IDF convierten textos en vectores numéricos que capturan similitudes semánticas y relaciones contextuales.
- **Herramientas de ingeniería automática:**  
  Librerías como Featuretools permiten la generación y combinación automática de características a partir de múltiples fuentes de datos.


### Aplicación en redes neuronales, procesamiento de datos y big data

#### Redes neuronales

- **Entrada y preprocesamiento:**  
  Las redes neuronales requieren datos numéricos normalizados y bien estructurados. La ingeniería de características es crucial para convertir datos categóricos a representaciones numéricas (por ejemplo, mediante one-hot encoding o embeddings) y para normalizar características continuas.  
- **Reducción de dimensionalidad:**  
  Con el uso de autoencoders o PCA, se pueden reducir las dimensiones de entrada, lo que facilita el entrenamiento de modelos profundos al disminuir el número de parámetros y mejorar la generalización.
- **Mejora en el rendimiento:**  
  Al proporcionar características de alta calidad, se pueden capturar relaciones complejas entre variables que las capas ocultas de una red neuronal podrán explotar para obtener mejores predicciones.

#### Procesamiento de Datos y Big Data

- **Escalabilidad:**  
  En entornos de big data, la transformación y limpieza de grandes volúmenes de datos requieren técnicas de ingeniería de características que sean escalables. Herramientas como Apache Spark ofrecen módulos para realizar estas transformaciones en paralelo, permitiendo trabajar con datasets de gran tamaño.
- **Integración y automatización:**  
  La automatización de la generación de características (por ejemplo, mediante Featuretools) es especialmente valiosa en contextos donde los datos provienen de múltiples fuentes o bases de datos relacionales.
- **Procesamiento en tiempo real:**  
  En aplicaciones de streaming o sistemas de recomendación, la ingeniería de características se aplica en tiempo real para transformar y normalizar datos antes de ser introducidos en modelos predictivos.


#### Código

#### 1. Lectura y selección de datos

```python
import pandas as pd
datos = pd.read_csv("adult.data", header=None, index_col=False,
                   names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                          'marital-status', 'occupation', 'relationship', 'race', 'gender',
                          'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                          'income'])
# Seleccionamos algunas columnas
datos = datos[['age', 'workclass', 'education', 'gender', 'hours-per-week','occupation', 'income']]
display(datos.head())
```

- **Objetivo:**  
  Se carga el archivo CSV que contiene los datos. Se especifica que no hay cabecera en el archivo y se asignan nombres a cada columna. Luego se seleccionan únicamente las columnas relevantes para el análisis.
- **Importancia:**  
  Seleccionar columnas específicas ayuda a focalizar el problema y reduce la complejidad del preprocesamiento.

#### 2. Verificación de datos categóricos

```python
print(datos.gender.value_counts())
```

- **Objetivo:**  
  Se utiliza `value_counts()` para identificar la frecuencia de cada valor presente en la columna `gender`. Esto es esencial para detectar posibles inconsistencias en los datos (por ejemplo, variaciones en la nomenclatura).
- **Importancia:**  
  Garantiza que la codificación posterior de las variables categóricas sea coherente.

#### 3. Codificación one-hot de variables categóricas

```python
print("Caracteristicas originales:\n", list(datos.columns), "\n")
datos_dummies = pd.get_dummies(datos)
print("Caracteristicas despues de  get_dummies:\n", list(datos_dummies.columns))
datos_dummies.head()
```

- **Objetivo:**  
  Se transforma el DataFrame original en uno donde las variables categóricas se codifican en variables dummy (0 o 1) mediante la función `get_dummies()` de pandas.
- **Consideración clave:**  
  Es fundamental asegurarse de que tanto el conjunto de entrenamiento como el de prueba tengan las mismas columnas después de la codificación para evitar errores en el modelado.

### 4. Conversión a matriz numérica y separación de variables

```python
# Seleccionar características (corrigiendo el error de ix)
caracteristica = datos_dummies.loc[:, 'age':'occupation_ Transport-moving']

# Extraemos matrices NumPy
X = caracteristica.values
y = datos_dummies['income_ >50K'].values

print("X.shape: {} y.shape: {}".format(X.shape, y.shape))
```

- **Objetivo:**  
  Se extraen las características independientes en una matriz **X** y la variable objetivo en **y**. Se hace uso del slicing en pandas, que incluye el límite superior, diferenciándose del comportamiento en NumPy.
- **Importancia:**  
  La conversión a matrices NumPy es necesaria para que las bibliotecas de machine learning, como scikit-learn, puedan procesar los datos.

#### 5. División de datos, escalado y entrenamiento del modelo

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Dividimos los datos en conjunto de entrenamiento y prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, random_state=0, test_size=0.2)

# Escalamos los datos para mejorar la estabilidad numérica
scaler = StandardScaler()
X_entrenamiento = scaler.fit_transform(X_entrenamiento)
X_prueba = scaler.transform(X_prueba)

# Definimos el modelo de regresión logística con más iteraciones y un solver adecuado
logreg = LogisticRegression(solver='saga', max_iter=1000, random_state=0)

# Entrenamos el modelo
logreg.fit(X_entrenamiento, y_entrenamiento)

# Evaluamos el modelo
puntuacion = logreg.score(X_prueba, y_prueba)
print("Puntuación del modelo en el conjunto de prueba: {:.2f}".format(puntuacion))
```

- **División de datos:**  
  Se utiliza `train_test_split` para separar el dataset en entrenamiento y prueba, garantizando que la evaluación se realice en datos no vistos.
- **Escalado:**  
  Se aplica la estandarización mediante `StandardScaler`, lo cual es crucial para algoritmos basados en gradientes y para evitar que las diferencias en magnitudes de las características afecten el entrenamiento.
- **Modelo de regresión logística:**  
  Se crea un modelo de regresión logística utilizando el solver `'saga'`, el cual es eficiente en datasets grandes y con regularización, y se ajusta el número de iteraciones para asegurar la convergencia.
- **Evaluación:**  
  La puntuación obtenida en el conjunto de prueba permite medir la efectividad del modelo. Es fundamental que la variable objetivo se haya separado correctamente para evitar la inclusión de datos irrelevantes en el entrenamiento.

