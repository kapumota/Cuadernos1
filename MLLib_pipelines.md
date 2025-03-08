### 1. ¿Qué son los pipelines de ML en SparkML?

En SparkML, un pipeline es una secuencia de etapas (stages) en las que cada etapa representa una transformación de datos o un modelo de aprendizaje automático. La idea central es encadenar procesos de preprocesamiento y modelado de manera sistemática y repetible. Cada pipeline consta de:
  
- **Transformadores:** Componentes que toman un DataFrame como entrada y devuelven otro DataFrame transformado. Por ejemplo, se usan para ensamblar características, escalar datos o transformar etiquetas.
- **Estimadores:** Algoritmos que aprenden a partir de los datos (por ejemplo, modelos de regresión o clasificación). Cuando se ajustan a los datos, producen un transformador (por ejemplo, un modelo entrenado).

El uso de pipelines permite que todo el flujo de trabajo de Machine Learning se defina de forma modular. Esto resulta en procesos de entrenamiento y validación más limpios y reproducibles, ya que cada paso se puede encapsular en una etapa única del pipeline. Además, se facilita el ajuste de hiperparámetros, la validación cruzada y la producción, ya que se trabaja con un único objeto que engloba todas las transformaciones y el modelo.

### 2. Pipeline de regresión con SparkML

En este ejemplo, se utiliza el conjunto de datos “mpg", que contiene información de automóviles (peso, potencia, desplazamiento del motor, etc.) y la variable objetivo es el consumo de combustible (MPG). Se crean tres etapas en el pipeline de regresión:

#### 2.1. Etapa 1: Ensamblaje de características con *VectorAssembler*

- **Objetivo:** Combinar varias columnas numéricas en una única columna de tipo vector que será utilizada por el modelo.  
- **Código:**

  ```python
  vectorAssembler = VectorAssembler(
      inputCols=["Weight", "Horsepower", "Engine Disp"],  # Columnas de entrada
      outputCol="features"  # Nombre de la columna resultante
  )
  ```

- **Explicación:**  
  El *VectorAssembler* toma las columnas “Weight", “Horsepower" y “Engine Disp" y las consolida en una sola columna llamada “features". Esto es fundamental en SparkML ya que la mayoría de los algoritmos requieren que las características se presenten como un vector numérico.

#### 2.2. Etapa 2: Escalado de características con *StandardScaler*

- **Objetivo:** Normalizar las características para que todas tengan una escala similar. Esto es útil para modelos que son sensibles a la magnitud de los datos.  
- **Código:**

  ```python
  scaler = StandardScaler(
      inputCol="features",       # Columna de entrada con el vector de características
      outputCol="scaledFeatures" # Columna de salida con las características escaladas
  )
  ```

- **Explicación:**  
  El *StandardScaler* toma el vector de características ensamblado y aplica una transformación para centrar y escalar los datos. Esto reduce problemas como la dominancia de una característica con valores más grandes y puede mejorar la convergencia del algoritmo de optimización en el modelo.

#### 2.3. Etapa 3: Modelo de regresión con *LinearRegression*

- **Objetivo:** Ajustar un modelo de regresión lineal que aprenda a predecir el consumo de combustible (MPG) a partir de las características escaladas.  
- **Código:**

  ```python
  lr = LinearRegression(
      featuresCol="scaledFeatures",  # Usamos las características escaladas
      labelCol="MPG"                 # La variable a predecir
  )
  ```

- **Explicación:**  
  Se crea una instancia de regresión lineal que utiliza la columna “scaledFeatures" para entrenar el modelo y “MPG" como la variable objetivo. La regresión lineal es un método estadístico que modela la relación lineal entre las variables independientes y la dependiente.

#### 2.4. Construcción y ajuste del pipeline

- **Construcción del pipeline:**  
  Una vez definidas todas las etapas, se construye el pipeline pasando las etapas en el orden en que deben ejecutarse.

  ```python
  pipeline = Pipeline(stages=[vectorAssembler, scaler, lr])
  ```

- **División de datos:**  
  Los datos se dividen en conjuntos de entrenamiento y prueba para evaluar el rendimiento del modelo.

  ```python
  (trainingData, testData) = mpg_data.randomSplit([0.7, 0.3], seed=42)
  ```

- **Ajuste del pipeline:**  
  El pipeline se entrena sobre los datos de entrenamiento. Durante este proceso, cada etapa se ajusta secuencialmente.

  ```python
  model = pipeline.fit(trainingData)
  ```

#### 2.5. Evaluación del Modelo de Regresión

- **Realización de predicciones:**  
  Una vez entrenado el pipeline, se generan predicciones para el conjunto de datos de prueba.

  ```python
  predictions = model.transform(testData)
  ```

- **Cálculo del error (RMSE):**  
  Se utiliza el *RegressionEvaluator* para calcular el error cuadrático medio (RMSE), que es una medida de la precisión del modelo.

  ```python
  evaluator = RegressionEvaluator(labelCol="MPG", predictionCol="prediction", metricName="rmse")
  rmse = evaluator.evaluate(predictions)
  print("Root Mean Squared Error (RMSE) =", rmse)
  ```

- **Explicación:**  
  El RMSE mide la magnitud del error en las predicciones del modelo. Un valor menor de RMSE indica que el modelo se ajusta mejor a los datos. Esta métrica es muy utilizada en problemas de regresión porque penaliza de forma cuadrática los errores grandes.

### 3. Pipeline de clasificación con PySpark

Para la tarea de clasificación se utiliza el conjunto de datos Iris. Este conjunto contiene mediciones de características de flores (por ejemplo, longitud y anchura del sépalo y pétalo) y una etiqueta que indica la especie de la flor. El proceso para construir un pipeline de clasificación es similar al de regresión, pero se agregan pasos específicos para manejar variables categóricas y se emplea un modelo de clasificación.

#### 3.1. Carga y exploración del conjunto de datos Iris

Primero, se carga el conjunto de datos Iris en un DataFrame de Spark. Se asume que el archivo CSV tiene encabezados y que Spark puede inferir los tipos de datos:

```python
# Cargar el conjunto de datos Iris
iris_data = spark.read.csv("iris.csv", header=True, inferSchema=True)
iris_data.printSchema()
iris_data.show(5)
```

- **Explicación:**  
  Con la función `spark.read.csv` se carga el archivo, se establece que el archivo posee encabezado y se infiere el esquema automáticamente. La inspección del esquema y la visualización de algunas filas ayudan a entender la estructura de los datos y a identificar las columnas que contienen las características y la etiqueta (por ejemplo, “species").

#### 3.2. Etapa 1: Ensamblaje de características

Se utiliza nuevamente el *VectorAssembler* para combinar las columnas que representan las características en un solo vector:

```python
vectorAssembler2 = VectorAssembler(
    inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"],  # Columnas de entrada
    outputCol="features"  # Columna resultante
)
```

- **Explicación:**  
  Este paso es similar al de la regresión. Se seleccionan las columnas relevantes (en este caso, las medidas de los sépalos y pétalos) y se ensamblan en una columna “features", que será la entrada para el modelo de clasificación.

#### 3.3. Etapa 2: Indexación de la etiqueta con *StringIndexer*

Dado que la variable "species" es categórica (por ejemplo, "setosa", "versicolor", "virginica"), es necesario transformarla en un valor numérico para que el algoritmo pueda procesarla. Para ello, se utiliza *StringIndexer*:

```python
labelIndexer = StringIndexer(
    inputCol="species",  # Columna categórica original
    outputCol="label"    # Columna de salida con valores numéricos
)
```

- **Explicación:**  
  *StringIndexer* asigna un índice numérico a cada categoría de la columna "species". Este paso es esencial en la mayoría de los algoritmos de clasificación en SparkML, ya que requieren que las etiquetas sean numéricas. La columna resultante "label" contendrá estos índices.

#### 3.4. Etapa 3: Modelo de clasificación con *LogisticRegression*

En este ejemplo se utiliza la regresión logística, un algoritmo de clasificación lineal, que es sencillo pero efectivo para problemas de clasificación multiclase:

```python
lr_classifier = LogisticRegression(
    featuresCol="features",  # Columna de características (ya ensamblada)
    labelCol="label"         # Columna de etiqueta (indexada)
)
```

- **Explicación:**  
  La regresión logística modela la probabilidad de que una instancia pertenezca a cada una de las clases. Aunque es un modelo lineal, puede ser muy efectivo en problemas de clasificación con fronteras de decisión lineales o cuando se preprocesan adecuadamente las características.

#### 3.5. Construcción del pipeline de clasificación

Se encadenan las etapas necesarias en un pipeline. En este caso, el pipeline para clasificación consta de tres etapas: ensamblaje de características, indexación de la etiqueta y el modelo de regresión logística.

```python
pipeline_classifier = Pipeline(stages=[vectorAssembler2, labelIndexer, lr_classifier])
```

- **Explicación:**  
  El pipeline se construye pasando las etapas en el orden correcto: primero se combinan las características, luego se transforma la etiqueta categórica en numérica y, finalmente, se entrena el modelo de clasificación.

#### 3.6. División, ajuste y evaluación del pipeline de clasificación

Al igual que en el caso de regresión, es importante dividir el conjunto de datos en entrenamiento y prueba para evaluar el desempeño del modelo.

```python
# División del conjunto de datos Iris en entrenamiento y prueba
(trainingData, testData) = iris_data.randomSplit([0.7, 0.3], seed=42)

# Ajuste del pipeline de clasificación con los datos de entrenamiento
model_classifier = pipeline_classifier.fit(trainingData)

# Realización de predicciones sobre el conjunto de prueba
predictions_classifier = model_classifier.transform(testData)
```

- **Explicación:**  
  Se utiliza el método `randomSplit` para dividir los datos de forma aleatoria en dos subconjuntos. Posteriormente, el pipeline se entrena sobre el conjunto de entrenamiento y se generan predicciones para el conjunto de prueba.

Para evaluar el modelo de clasificación se utiliza el *MulticlassClassificationEvaluator*. En este caso, se puede emplear la métrica de exactitud (accuracy):

```python
evaluator_classifier = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction", 
    metricName="accuracy"
)

accuracy = evaluator_classifier.evaluate(predictions_classifier)
print("Accuracy =", accuracy)
```

- **Explicación:**  
  El evaluador compara las etiquetas reales con las predichas y calcula la exactitud, que es la proporción de predicciones correctas. Este valor es crucial para entender qué tan bien el modelo clasifica las instancias del conjunto de prueba.

### 4. Flujo de trabajo y componentes clave

La implementación de pipelines en SparkML sigue una serie de pasos estructurados que permiten manejar todo el flujo de trabajo de Machine Learning de manera ordenada:

#### 4.1. Configuración e importación de librerías

Antes de definir los pipelines es necesario configurar el entorno y cargar las librerías esenciales. Por ejemplo:

```python
# Supresión de advertencias para mantener el output limpio
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# Inicialización de Spark mediante findspark
import findspark
findspark.init()

# Importación de clases y funciones para modelos, transformaciones y evaluación
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
```

- **Explicación:**  
  Se configuran advertencias, se inicializa Spark y se importan los módulos necesarios para la creación y evaluación de modelos. Esto garantiza que todos los componentes estén disponibles para construir y ejecutar los pipelines.

#### 4.2. Creación de la SparkSession

La *SparkSession* es la entrada principal para trabajar con Spark. Se crea una sesión con un nombre identificativo:

```python
spark = SparkSession.builder.appName("Ejemplo de un pipeline de ML").getOrCreate()
```

- **Explicación:**  
  La SparkSession gestiona el contexto de ejecución y permite leer datos, ejecutar transformaciones y entrenar modelos. Es esencial para cualquier aplicación en Spark.

#### 4.3. Carga y exploración de datos

El primer paso en cualquier proyecto de Machine Learning es la carga y exploración de datos. Por ejemplo, en el caso del conjunto "mpg":

```python
mpg_data = spark.read.csv("mpg.csv", header=True, inferSchema=True)
mpg_data.printSchema()
mpg_data.show(5)
```

- **Explicación:**  
  Se leen los datos del archivo CSV, se imprime el esquema para verificar los tipos de datos y se muestran algunas filas para entender la distribución y calidad de los datos.


### 5. Detalle de cada componente del pipeline

#### 5.1. *VectorAssembler*

- **Función principal:**  
  Este transformador combina varias columnas de entrada en una única columna de vector. Es crucial para preparar datos de entrada para los modelos ML en Spark, ya que la mayoría de ellos requieren que las características se encuentren en forma vectorial.

- **Uso en regresión y clasificación:**  
  En ambos ejemplos (mpg e iris), *VectorAssembler* se utiliza para agrupar las columnas numéricas relevantes. En el caso del conjunto de iris, se combinan medidas de sépalo y pétalo, mientras que en el conjunto de mpg se combinan variables relacionadas con las características del vehículo.

#### 5.2. *StandardScaler*

- **Función principal:**  
  Normaliza los datos, lo cual es particularmente útil en algoritmos de optimización que son sensibles a la escala de las variables.  
- **Aplicación en el pipeline de regresión:**  
  Se aplica después del ensamblaje de las características para garantizar que todas las variables tengan media cero y varianza uno, mejorando así el rendimiento del modelo.

#### 5.3. *StringIndexer*

- **Función principal:**  
  Transforma variables categóricas (como “species" en el conjunto Iris) en índices numéricos.  
- **Importancia en clasificación:**  
  Permite convertir etiquetas textuales en valores numéricos que los algoritmos de clasificación pueden interpretar correctamente. Esto es fundamental para el modelo de regresión logística empleado en el pipeline de clasificación.

#### 5.4. Modelos: *LinearRegression* y *LogisticRegression*

- **LinearRegression:**  
  Se utiliza para predecir una variable continua (en este caso, MPG). El modelo aprende una relación lineal entre las características (después de ser escaladas) y la variable objetivo.

- **LogisticRegression:**  
  Es un modelo de clasificación lineal que estima probabilidades para la pertenencia a cada clase. Se utiliza en el ejemplo del conjunto Iris para clasificar las especies de flores basándose en las características medidas.


### 6. Ejecución y Evaluación de los Pipelines

### 6.1. Ajuste del Modelo

Para ambos casos, el pipeline se ajusta a los datos de entrenamiento. Durante este ajuste, cada etapa se entrena secuencialmente. Esto significa que:
  
- En el pipeline de regresión, *VectorAssembler* y *StandardScaler* se aplican a los datos para generar las columnas “features" y “scaledFeatures", y luego el modelo de regresión aprende a predecir “MPG".
  
- En el pipeline de clasificación, el ensamblaje de características y la indexación de etiquetas preparan los datos para que el clasificador (regresión logística) pueda aprender a distinguir entre las diferentes especies de iris.

### 6.2. Predicción y Evaluación

- **Predicción:**  
  Tras el ajuste, se generan predicciones sobre el conjunto de prueba utilizando el método `transform`. Este método aplica todas las transformaciones definidas en el pipeline, incluidas las etapas de preprocesamiento y la predicción final del modelo.

- **Evaluación del Modelo de Regresión:**  
  Se utiliza el *RegressionEvaluator* para calcular el error cuadrático medio (RMSE). Esta métrica es especialmente útil para cuantificar el error de las predicciones en problemas de regresión.

- **Evaluación del Modelo de Clasificación:**  
  Para la clasificación, se emplea el *MulticlassClassificationEvaluator* con la métrica “accuracy", que mide el porcentaje de instancias clasificadas correctamente. Esta evaluación es fundamental para entender la eficacia del modelo en distinguir entre múltiples clases.

---

## 7. Código Completo de Ejemplo para Regresión y Clasificación

A continuación se presenta un ejemplo completo que integra todos los componentes explicados. Se incluye la configuración inicial, la carga de datos, la construcción de pipelines, el ajuste, la predicción y la evaluación para ambos casos.

### 7.1. Pipeline de Regresión

```python
# Configuración inicial y librerías
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# Creación de la SparkSession
spark = SparkSession.builder.appName("Ejemplo de pipeline de regresión").getOrCreate()

# Cargar el conjunto de datos mpg
mpg_data = spark.read.csv("mpg.csv", header=True, inferSchema=True)
mpg_data.printSchema()
mpg_data.show(5)

# Definición de las etapas del pipeline
vectorAssembler = VectorAssembler(
    inputCols=["Weight", "Horsepower", "Engine Disp"],
    outputCol="features"
)
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures"
)
lr = LinearRegression(
    featuresCol="scaledFeatures",
    labelCol="MPG"
)

# Construcción del pipeline
pipeline = Pipeline(stages=[vectorAssembler, scaler, lr])

# División de datos en entrenamiento y prueba
(trainingData, testData) = mpg_data.randomSplit([0.7, 0.3], seed=42)

# Ajuste del pipeline
model = pipeline.fit(trainingData)

# Predicción en el conjunto de prueba
predictions = model.transform(testData)

# Evaluación del modelo con RMSE
evaluator = RegressionEvaluator(labelCol="MPG", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) =", rmse)

# Finalizar la sesión de Spark
spark.stop()
```

### 7.2. Pipeline de Clasificación

```python
# Configuración inicial y librerías
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Creación de la SparkSession
spark = SparkSession.builder.appName("Ejemplo de pipeline de clasificación").getOrCreate()

# Cargar el conjunto de datos Iris
iris_data = spark.read.csv("iris.csv", header=True, inferSchema=True)
iris_data.printSchema()
iris_data.show(5)

# Definir las etapas del pipeline para clasificación
vectorAssembler2 = VectorAssembler(
    inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    outputCol="features"
)
labelIndexer = StringIndexer(
    inputCol="species",
    outputCol="label"
)
lr_classifier = LogisticRegression(
    featuresCol="features",
    labelCol="label"
)

# Construir el pipeline de clasificación
pipeline_classifier = Pipeline(stages=[vectorAssembler2, labelIndexer, lr_classifier])

# División de los datos en entrenamiento y prueba
(trainingData, testData) = iris_data.randomSplit([0.7, 0.3], seed=42)

# Ajuste del pipeline con los datos de entrenamiento
model_classifier = pipeline_classifier.fit(trainingData)

# Predicción en el conjunto de prueba
predictions_classifier = model_classifier.transform(testData)

# Evaluación del modelo utilizando exactitud (accuracy)
evaluator_classifier = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction", 
    metricName="accuracy"
)
accuracy = evaluator_classifier.evaluate(predictions_classifier)
print("Accuracy =", accuracy)

# Finalizar la sesión de Spark
spark.stop()
```

---

## 8. Aspectos Técnicos y Detalles Adicionales

### 8.1. Flujo de Ejecución del Pipeline

Cuando se invoca el método `fit` sobre un pipeline, SparkML procede de la siguiente manera:
  
- **Aplicación Secuencial de Transformaciones:**  
  Cada etapa (por ejemplo, el ensamblaje de características, el escalado, la indexación) se aplica en orden. Esto garantiza que las transformaciones se encadenen y que cada etapa trabaje sobre el resultado de la etapa anterior.
  
- **Ajuste del Modelo:**  
  Una vez transformados los datos, el estimador (ya sea un modelo de regresión o clasificación) se ajusta a los datos transformados. El objeto resultante es un modelo entrenado que incluye la lógica para aplicar las transformaciones y luego hacer predicciones.

### 8.2. Evaluadores y Métricas

- **RegressionEvaluator:**  
  Permite evaluar modelos de regresión utilizando métricas como RMSE, MAE o R². La elección de la métrica depende del problema y de la interpretación que se quiera dar al error.

- **MulticlassClassificationEvaluator:**  
  Evalúa modelos de clasificación multiclase utilizando métricas como la exactitud (accuracy), F1-score, entre otras. La exactitud es una medida directa de la proporción de predicciones correctas.

### 8.3. Ventajas de Usar Pipelines

- **Modularidad y Reusabilidad:**  
  Cada etapa del pipeline se puede desarrollar, probar y reutilizar de forma independiente. Esto permite modificar o mejorar partes del proceso sin alterar el flujo completo.

- **Facilidad de Mantenimiento:**  
  Al encapsular el preprocesamiento y el modelado en un solo objeto, se simplifica el mantenimiento del código, lo que es especialmente útil en entornos de producción.

- **Escalabilidad:**  
  SparkML está diseñado para trabajar con grandes volúmenes de datos. La capacidad de definir pipelines escalables permite aplicar el mismo flujo de trabajo tanto en conjuntos de datos pequeños como en escenarios de big data.

- **Reproducibilidad:**  
  Los pipelines aseguran que el mismo proceso de transformación se aplique a nuevos datos. Esto es vital para la consistencia en la producción y para la replicación de resultados.

### 8.4. Consideraciones sobre el Preprocesamiento

- **Selección de Características:**  
  La elección de las columnas a incluir en el ensamblaje es crítica. En el ejemplo de “mpg" se seleccionan características relevantes para predecir el consumo de combustible. En el conjunto Iris se utilizan todas las medidas disponibles, ya que cada una contribuye a la clasificación de la especie.

- **Normalización y Escalado:**  
  Transformaciones como la normalización ayudan a estabilizar el proceso de entrenamiento, especialmente en algoritmos que se basan en distancias o que son sensibles a la escala de las variables.

- **Manejo de Variables Categóricas:**  
  Transformar variables categóricas en índices numéricos (por medio de *StringIndexer*) es un paso indispensable en pipelines de clasificación. Sin este paso, los algoritmos no podrían procesar etiquetas textuales.

### 8.5. Ajuste y Validación del Modelo

- **División de Datos:**  
  Dividir el conjunto de datos en entrenamiento y prueba (u otros esquemas como validación cruzada) es fundamental para evaluar la capacidad del modelo para generalizar a datos no vistos.

- **Ajuste de Hiperparámetros:**  
  Aunque en los ejemplos se usan configuraciones predeterminadas para los modelos, en aplicaciones reales es común utilizar técnicas de búsqueda en cuadrícula (grid search) o validación cruzada para encontrar los parámetros óptimos.

- **Interpretación de Resultados:**  
  Las métricas de evaluación (como RMSE en regresión y accuracy en clasificación) proporcionan una idea del rendimiento del modelo, pero deben ser interpretadas en el contexto del dominio y de las necesidades específicas de la aplicación.

---

## 9. Consideraciones Finales 

La implementación de pipelines en SparkML representa una práctica robusta y escalable para la creación de modelos de Machine Learning. Tanto el pipeline de regresión como el de clasificación presentados muestran cómo integrar múltiples transformaciones y modelos en un solo flujo de trabajo, facilitando la experimentación y la producción.

- La modularidad de las etapas permite modificar o agregar pasos sin tener que reestructurar completamente el código.
- La capacidad de dividir los datos, ajustar modelos y evaluarlos de forma integrada es una ventaja clave en entornos de big data.
- El uso de transformaciones específicas para cada tipo de dato (numérico o categórico) garantiza que los modelos reciban la entrada en el formato adecuado.

Este enfoque se vuelve especialmente valioso en proyectos reales, donde la calidad y la preparación de los datos tienen un impacto directo en la precisión y robustez de los modelos de Machine Learning. Además, la integración de evaluadores permite medir de forma objetiva el rendimiento del modelo y ajustar estrategias de preprocesamiento o elección de algoritmos.

Por último, la flexibilidad que ofrecen los pipelines en SparkML permite que se extienda la metodología a otros algoritmos, técnicas de ensamblaje y métodos de evaluación, adaptándose a las necesidades específicas de cada aplicación. Los ejemplos proporcionados aquí son solo una introducción a lo que se puede lograr utilizando PySpark en proyectos de Machine Learning, abriendo la puerta a implementaciones más complejas y robustas en entornos distribuidos.

