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

#### Procesamiento de datos y big data

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

#### 4. Conversión a matriz numérica y separación de variables

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

#### Representación de variables categóricas con números

En muchos conjuntos de datos, las variables categóricas se pueden codificar de dos formas principales:  
- **Como cadenas de texto:**  
  - Permiten identificar de forma explícita la categoría (por ejemplo, `"Private"`, `"Government"`, etc.).  
  - Reducen la ambigüedad, ya que cada categoría se representa con una etiqueta legible.  
- **Como números enteros:**  
  - Esta representación es común para optimizar el almacenamiento o porque los datos se recopilan mediante cuestionarios, donde cada opción se asigna un número (por ejemplo, `0`, `1`, `2`, …).  
  - El problema surge cuando, a simple vista, no se distingue si la variable debe tratarse como numérica continua o como categórica. En el caso de que se trate de categorías discretas, se requiere realizar una transformación para evitar que se interprete una relación de orden o magnitud inexistente.

#### ¿Cuándo tratar una variable entera como categórica?

- **Sin orden lógico:**  
  Por ejemplo, en la columna `workclass` donde `0` podría representar `"Private Employee"` y `1` representar `"Government Employee"`. Aunque sean números, no existe una relación de magnitud entre ellos. En estos casos, es preferible tratarlas como variables categóricas y aplicar técnicas de codificación como *one-hot encoding*.  
- **Con escala ordinal:**  
  Por ejemplo, en calificaciones de estrellas de 1 a 5, donde sí existe un orden natural. La decisión de tratarlas como continuas o categóricas dependerá del modelo y del contexto.

#### Métodos de codificación: `pandas.get_dummies()` vs `OneHotEncoder`

#### pandas.get_dummies()

- **Funcionalidad:**  
  Esta función transforma automáticamente las columnas de tipo `object` (o de tipo categórico) en variables dummy, es decir, columnas binarias que indican la presencia o ausencia de una categoría.
- **Limitación:**  
  Si la columna está compuesta por números enteros, `get_dummies()` los interpreta como datos continuos y no genera las variables dummy, lo que puede resultar en una representación inadecuada.

#### OneHotEncoder de scikit‑learn

- **Funcionalidad:**  
  Permite especificar explícitamente qué columnas son categóricas, incluso si están representadas como números. Es más flexible porque se puede ajustar para trabajar con variables que ya sean numéricas, creando una codificación one-hot adecuada.
- **Uso en redes neuronales:**  
  Las redes neuronales requieren entradas numéricas y, en muchos casos, la codificación one-hot es la forma preferida de representar variables categóricas. Esto garantiza que la red reciba vectores binarios que representan la presencia de una categoría, lo que evita que la red asuma una relación ordinal cuando no existe.

#### Ejemplo en Python: Comparando ambas aproximaciones

En el primer bloque de código se crea un DataFrame con dos columnas:

- **`workclass_str`:** Contiene valores categóricos como cadenas de texto.
- **`workclass_int`:** Contiene la misma información codificada numéricamente.

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Creamos un DataFrame con una columna categórica como string y otra como entero
df = pd.DataFrame({
    'workclass_str': ['Private', 'Self Employed', 'Government'],
    'workclass_int': [0, 1, 2]  # Codificación numérica
})

# Aplicamos get_dummies en la columna string
df_dummies = pd.get_dummies(df, columns=['workclass_str'])

# Usamos OneHotEncoder de sklearn para la columna numérica
encoder = OneHotEncoder(sparse=False)
encoded_values = encoder.fit_transform(df[['workclass_int']])
encoded_df = pd.DataFrame(encoded_values, columns=encoder.get_feature_names_out(['workclass_int']))

# Concatenamos los resultados
final_df = pd.concat([df_dummies, encoded_df], axis=1)

# Mostramos el resultado
print(final_df)
```

- **Explicación:**  
  - Se crea el DataFrame original con dos tipos de representación para la misma variable.  
  - Se transforma la columna de cadenas mediante `get_dummies()`, generando variables dummy cuyos nombres reflejan cada categoría de texto.  
  - La columna numérica se transforma con `OneHotEncoder`, que reconoce los valores únicos (0, 1, 2) y los convierte en columnas binarias.  
  - Finalmente, se concatenan ambos resultados para obtener una representación completa en la que ambas columnas se tratan como categóricas.

En el segundo ejemplo se muestra otro DataFrame (`demo_df`) en el que se ve claramente que `get_dummies` por defecto solo codifica la columna con cadenas, y para incluir la columna numérica se debe convertir a tipo cadena o especificar las columnas a codificar:

```python
# creamos un DataFrame con una característica entera y una característica de cadena categórica
demo_df = pd.DataFrame({'Caracteristica entera': [0, 1, 2, 1],
                        'Caracteristica categorica': ['socks', 'fox', 'socks', 'box']})
display(demo_df)
```

Si se utiliza `pd.get_dummies(demo_df)` se observa que solo se codifica la columna de tipo cadena. Para incluir la columna numérica se puede convertir:

```python
demo_df['Caracteristica entera'] = demo_df['Caracteristica entera'].astype(str)
pd.get_dummies(demo_df, columns=['Caracteristica entera', 'Caracteristica categorica'])
```

- **Importancia en redes neuronales:**  
  Al transformar variables categóricas en vectores one-hot, se evita que la red neuronal confunda una variable discreta con una continua. Esto es esencial ya que las redes neuronales procesan entradas numéricas y, en este caso, cada neurona de entrada recibirá 0 o 1, lo que permite que la red aprenda patrones específicos de cada categoría sin asumir relaciones de orden implícitas.

#### Aplicación de binning y transformación en datos continuos

Una técnica adicional importante es la transformación de variables continuas en variables categóricas mediante *binning*. Esto consiste en dividir el rango de valores continuos en intervalos (o *bins*), y luego asignar a cada dato el índice del intervalo al que pertenece.

#### Ejemplo con el conjunto de datos `wave`

Se crea un conjunto de datos sintético mediante la función `hacer_wave`, que genera una característica continua y una salida que es una combinación de una función seno y una componente lineal. Posteriormente se define un conjunto de intervalos:

```python
import numpy as np
def hacer_wave(n_muestras=100):
    rnd = np.random.RandomState(42)
    x = rnd.uniform(-3, 3, size=n_muestras)
    y_no_ruido = (np.sin(4 * x) + x)
    y = (y_no_ruido + rnd.normal(size=len(x))) / 2
    return x.reshape(-1, 1), y

# Generamos los datos
X, y = hacer_wave(n_muestras=100)
```

Se definen 10 *bins* mediante:

```python
bins = np.linspace(-3, 3, 11)
print("bins: {}".format(bins))
```

- **Uso de np.digitize:**  
  La función `np.digitize` asigna a cada valor de `X` el índice del *bin* en el que se encuentra:

```python
bin_pertenece = np.digitize(X, bins=bins)
print("\nPuntos de datos:\n", X[:5])
print("\n*Bin* al que pertenecen los puntos de datos:\n", bin_pertenece[:5])
```

Posteriormente, se transforma la variable categórica (representada por el índice del bin) en una codificación one-hot utilizando `OneHotEncoder`:

```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(bin_pertenece)
X_binned = encoder.transform(bin_pertenece)
print(X_binned[:5])
print("Dimensión de X_binned: {}".format(X_binned.shape))
```

- **Interpretación:**  
  Cada punto de datos se representa mediante un vector en el que solo la entrada correspondiente al bin al que pertenece es 1, y el resto son 0. Esto permite que se utilice la misma lógica de codificación para transformar variables continuas en categóricas y modelarlas con algoritmos que se benefician de una representación segmentada.


#### Integración de técnicas en modelos y redes neuronales

#### Uso directo en redes neuronales

- **Codificación One-Hot:**  
  Las redes neuronales, especialmente las redes profundas, requieren vectores de entrada numéricos. La representación one-hot de variables categóricas es común en arquitecturas que trabajan con datos tabulares. Sin embargo, cuando la cardinalidad es muy alta (por ejemplo, en NLP o en sistemas de recomendación), se pueden emplear técnicas de *embeddings* que convierten cada categoría en un vector denso de menor dimensión.  
- **Normalización y escalado:**  
  Las características continuas deben ser escaladas o normalizadas (por ejemplo, con `StandardScaler` o `MinMaxScaler`) para asegurar que la red no se vea afectada por escalas numéricas muy dispares.  
- **Combinación de características:**  
  La posibilidad de incluir características de interacción (por ejemplo, el producto entre el indicador del bin y la característica original) permite a la red neuronal capturar relaciones más complejas sin necesidad de una ingeniería manual excesiva. En arquitecturas modernas, algunas de estas interacciones se pueden aprender de forma implícita, pero la ingeniería de características puede mejorar la convergencia y la precisión del modelo.

#### Expansión con características de interacción y polinomiales

El código muestra dos enfoques adicionales para enriquecer la representación:

1. **Interacciones (Producto de características):**

   Se combina la representación one-hot del bin (`X_binned`) con la característica original `X` multiplicada por el indicador, utilizando `np.hstack`:

   ```python
   X_producto = np.hstack([X_binned, X * X_binned])
   print(X_producto.shape)
   ```

   - **Concepto:**  
     Esto crea un conjunto de datos con 20 características, donde para cada bin se incluye tanto un indicador (que aporta el desplazamiento) como el producto con la característica original (que aporta la pendiente).  
   - **Aplicación en redes neuronales:**  
     Aunque en redes neuronales se pueden aprender relaciones no lineales mediante múltiples capas, la inclusión de características de interacción explícitas puede ayudar a modelar relaciones locales en el espacio de la entrada, haciendo que la red sea más sensible a cambios específicos en ciertos intervalos.

2. **Características polinómicas:**

   Se utilizan las transformaciones polinómicas para expandir la variable original `X` hasta un cierto grado (por ejemplo, grado 10):

   ```python
   from sklearn.preprocessing import PolynomialFeatures
   polinomio = PolynomialFeatures(degree=10, include_bias=False)
   polinomio.fit(X)
   X_polinomio = polinomio.transform(X)
   print("Dimension de X_polinomio: {}".format(X_polinomio.shape))
   print("Nombres de características polinomiales:\n{}".format(polinomio.get_feature_names_out()))
   ```

   - **Concepto:**  
     Las características polinómicas permiten que un modelo lineal se comporte de forma no lineal, al incluir términos elevados (cuadráticos, cúbicos, etc.) de la variable original.  
   - **Relevancia en redes neuronales:**  
     Aunque las redes neuronales tienen la capacidad de aprender funciones complejas, la incorporación de características polinómicas puede ser útil en arquitecturas más simples o cuando se desea forzar la red a capturar ciertos patrones conocidos en el dominio de la aplicación. Además, en problemas de regresión, este tipo de ingeniería de características puede servir de base para comparaciones con modelos no lineales.

### Comparación de modelos basados en transformaciones

El ejemplo final muestra cómo distintos modelos se comportan cuando se aplican estas transformaciones:

- **Modelo de regresión lineal y árboles de decisión con datos agrupados (binned):**  
  Se observa que, al transformar la característica continua en variables categóricas mediante *binning* y codificación one-hot, tanto un modelo lineal como un árbol de decisión realizan predicciones constantes dentro de cada contenedor. Sin embargo, al incluir la variable original o interacciones, el modelo lineal puede ganar en flexibilidad, logrando diferentes pendientes para cada bin. Esto se visualiza en los gráficos donde se dibujan líneas verticales que marcan los límites de los bins.
  
- **Uso de un SVM con Kernel RBF:**  
  Se muestra cómo un modelo basado en SVM, que utiliza un kernel RBF, es capaz de aprender una función compleja sin necesidad de transformar explícitamente la característica. Esto resalta que algunos modelos pueden aprender representaciones no lineales directamente de los datos originales, aunque la ingeniería de características sigue siendo muy útil para simplificar la tarea de aprendizaje y mejorar la interpretabilidad del modelo.

A continuación se ofrece una explicación detallada del código y de los conceptos presentados, con un énfasis especial en cómo se pueden aplicar y adaptar estas técnicas al ámbito de las redes neuronales.

### Ejemplo

En el procesamiento de datos para modelos de machine learning, la calidad de las características (features) es fundamental. En el caso de las redes neuronales, la forma en que se presentan los datos al modelo afecta tanto la velocidad de convergencia durante el entrenamiento como la capacidad de generalización. Las transformaciones de características, que incluyen el escalado, la generación de interacciones y términos polinómicos, son herramientas muy poderosas. Aunque las redes neuronales profundas son capaces de aprender representaciones complejas a partir de datos sin procesar, en muchos escenarios –especialmente cuando se trabaja con conjuntos de datos de tamaño moderado o cuando se desea aprovechar conocimiento previo del dominio– la ingeniería de características explícita puede mejorar significativamente el rendimiento del modelo.

En el ejemplo que se presenta, se utiliza el conjunto de datos de California Housing para demostrar cómo se pueden construir nuevas características a partir de las originales y cómo estas transformaciones influyen en el rendimiento del modelo. Además, se ilustra la aplicación de transformaciones no lineales univariadas en un conjunto de datos de recuentos, lo cual es muy relevante en contextos en los que la distribución de las variables es asimétrica o presenta valores extremos.


#### 1. Carga y escalado de datos con MinMaxScaler

El primer bloque de código se encarga de cargar el conjunto de datos de California Housing y de reescalar las características para que todas se encuentren en un rango común (usualmente entre 0 y 1). Este paso es crucial para redes neuronales, ya que:

- **Optimiza la convergencia:**  
  Los algoritmos de optimización basados en gradiente (como Adam, SGD, etc.) funcionan mejor cuando las entradas están en una escala uniforme.  
- **Evita problemas numéricos:**  
  Diferencias muy grandes en la escala pueden llevar a gradientes desproporcionados que afecten el entrenamiento.

El código es el siguiente:

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Cargar el conjunto de datos de California Housing
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Dividir los datos en conjuntos de entrenamiento y prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, random_state=0)

# Reescalar los datos
escalador = MinMaxScaler()
X_entrenamiento_escalado = escalador.fit_transform(X_entrenamiento)
X_prueba_escalado = escalador.transform(X_prueba)
```

**Explicación:**  
- Se obtiene el dataset utilizando la función `fetch_california_housing()`.
- Se separan los datos en entrenamiento y prueba con `train_test_split()`.
- Se reescalan las características mediante `MinMaxScaler()`, transformándolas a un rango [0, 1].  
En el contexto de redes neuronales, es común utilizar escaladores como MinMaxScaler o StandardScaler (para normalizar a media 0 y varianza 1), dependiendo de la naturaleza de los datos y del modelo utilizado.


#### 2. Extracción de características polinomiales e interacciones

Una vez que las características están escaladas, se procede a expandir el espacio de características utilizando términos polinomiales de grado 2. Esto significa que se generan tanto los términos individuales como las interacciones entre características. El código utilizado es:

```python
from sklearn.preprocessing import PolynomialFeatures

# Se generan características polinomiales (incluyendo interacciones) hasta el grado 2
polinomio = PolynomialFeatures(degree=2).fit(X_entrenamiento_escalado)
X_entrenamiento_polinomio = polinomio.transform(X_entrenamiento_escalado)
X_prueba_polinomio = polinomio.transform(X_prueba_escalado)

print("Dimensión X_entrenamiento: {}".format(X_entrenamiento.shape))
print("Dimensión X_entrenamiento_polinomio: {}".format(X_entrenamiento_polinomio.shape))
```

**Conceptos clave:**
- **Características polinomiales:**  
  Se incluyen términos de segundo orden que pueden capturar relaciones no lineales entre las variables originales. Por ejemplo, si se tiene una característica $x$, se añade $x^2$ y, si existen dos características $x_1$ y $x_2$, se añade el término de interacción $x_1 \times x_2$.
- **Interacciones:**  
  Permiten que el modelo capte cómo la combinación de dos variables puede influir en la variable objetivo.  
- **Importancia en redes neuronales:**  
  Aunque las redes profundas pueden, en teoría, aprender interacciones de forma implícita a través de múltiples capas, en redes neuronales más superficiales o cuando se dispone de pocos datos, la incorporación explícita de estas interacciones puede facilitar el aprendizaje de patrones complejos. Además, al transformar las entradas, se puede mejorar la interpretación y la estabilidad numérica del entrenamiento.

La función `get_feature_names_out()` permite conocer la correspondencia exacta entre las características originales y las generadas:

```python
print("Nombres de caracteristicas polinomiales:\n{}".format(polinomio.get_feature_names_out()))
```

Esto es útil para comprender qué combinaciones se están incluyendo, lo que puede servir para análisis posteriores o para aplicar técnicas de reducción de dimensionalidad si la expansión resulta excesiva.

#### 3. Comparación del rendimiento de modelos: Ridge vs. RandomForest

La idea es comparar cómo la incorporación de interacciones y términos polinómicos afecta el rendimiento de dos modelos distintos: un modelo lineal regularizado (Ridge) y un modelo basado en ensambles (RandomForest). Se realiza lo siguiente:

#### Modelo Ridge

```python
from sklearn.linear_model import Ridge

# Modelo Ridge sin interacciones
ridge = Ridge().fit(X_entrenamiento_escalado, y_entrenamiento)
print("Puntuación sin interacciones: {:.3f}".format(ridge.score(X_prueba_escalado, y_prueba)))

# Modelo Ridge con interacciones
ridge = Ridge().fit(X_entrenamiento_polinomio, y_entrenamiento)
print("Puntuación con interacciones: {:.3f}".format(ridge.score(X_prueba_polinomio, y_prueba)))
```

**Observaciones:**
- En modelos lineales, la incorporación de interacciones y términos polinómicos suele mejorar la capacidad del modelo para capturar relaciones no lineales.
- Para redes neuronales, especialmente en arquitecturas menos profundas, añadir estas transformaciones puede ayudar a reducir la complejidad que la red debe aprender, facilitando una convergencia más rápida y, en algunos casos, un mejor rendimiento.

#### Modelo RandomForest

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100).fit(X_entrenamiento_escalado, y_entrenamiento)
print("Puntuación sin interacciones: {:.3f}".format(rf.score(X_prueba_escalado, y_prueba)))

rf = RandomForestRegressor(n_estimators=100).fit(X_entrenamiento_polinomio, y_entrenamiento)
print("Puntuación con interacciones: {:.3f}".format(rf.score(X_prueba_polinomio, y_prueba)))
```

**Observaciones:**
- Los modelos basados en árboles (como RandomForest) capturan interacciones y no requieren necesariamente la transformación polinómica previa, ya que pueden aprender divisiones no lineales de manera automática.
- En algunos casos, agregar características polinomiales a un modelo de bosque aleatorio puede incluso disminuir ligeramente el rendimiento, ya que el modelo ya es suficientemente flexible y la expansión del espacio de características puede introducir ruido o redundancia.

**Orientación a redes neuronales:**  
- En el contexto de redes neuronales, la incorporación de características polinomiales o de interacción es un arma de doble filo. En redes profundas con suficiente capacidad, la red puede aprender estas relaciones de forma interna. Sin embargo, en arquitecturas más simples o cuando se dispone de pocos datos, la ingeniería de características explícita puede ayudar a mejorar el rendimiento.
- Además, la presencia de muchas características generadas (especialmente en grados altos) puede incrementar el riesgo de sobreajuste, por lo que es importante considerar técnicas de regularización y métodos de reducción de dimensionalidad.

#### 4. Transformaciones no lineales univariadas

Cuando las características presentan distribuciones muy asimétricas o están sesgadas, las transformaciones no lineales pueden ayudar a que la distribución se asemeje más a la gaussiana, lo que facilita el aprendizaje de modelos lineales y redes neuronales. En el ejemplo se utilizan transformaciones logarítmicas para datos de recuentos.

#### Generación de datos artificiales de recuento

```python
import numpy as np
import matplotlib.pyplot as plt

rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000, 3))
w = rnd.normal(size=3)

# Se generan recuentos (valores enteros positivos) a partir de una transformación exponencial
X = rnd.poisson(10 * np.exp(X_org))
y = np.dot(X_org, w)
```

**Contexto y propósito:**
- Se generan datos artificiales en los que las características son enteras y representan recuentos. Los recuentos suelen tener distribuciones sesgadas y pueden presentar una alta varianza.
- En redes neuronales, entrenar con datos con distribuciones muy asimétricas puede dificultar la convergencia, ya que la red es sensible a la escala y distribución de cada variable.

#### Visualización y análisis de la distribución

```python
print("Número de apariciones de la primera característica:\n{}".format(np.bincount(X[:, 0])))
```

Posteriormente se visualiza la frecuencia de aparición de cada valor:

```python
bins_val = np.bincount(X[:, 0])
plt.bar(range(len(bins_val)), bins_val, color='grey')
plt.ylabel("Número de apariciones")
plt.xlabel("Valores")
plt.show()
```

**Interpretación:**
- El histograma permite observar que algunos valores tienen una frecuencia muy alta mientras que otros aparecen con poca frecuencia, lo cual es típico en datos de recuentos.

#### Ajuste de un modelo Ridge antes y después de la transformación logarítmica

Sin transformación:

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, random_state=0)
puntuacion = Ridge().fit(X_entrenamiento, y_entrenamiento).score(X_prueba, y_prueba)
print("Puntuación del conjunto de prueba (sin transformación): {:.3f}".format(puntuacion))
```

Dado que la función logaritmo no está definida para 0, se aplica la transformación logarítmica ajustada (log(X + 1)):

```python
X_entrenamiento_log = np.log(X_entrenamiento + 1)
X_prueba_log = np.log(X_prueba + 1)
```

Se visualiza el histograma de la primera característica transformada:

```python
plt.hist(X_entrenamiento_log[:, 0], bins=25, color='gray')
plt.ylabel("Número de apariciones")
plt.xlabel("Valores transformados")
plt.show()
```

Finalmente, se vuelve a ajustar el modelo Ridge sobre los datos transformados:

```python
puntuacion = Ridge().fit(X_entrenamiento_log, y_entrenamiento).score(X_prueba_log, y_prueba)
print("Puntuación del conjunto de prueba (con transformación logarítmica): {:.3f}".format(puntuacion))
```

**Relevancia en redes neuronales:**
- Las transformaciones logarítmicas, exponenciales, o funciones trigonométricas (como sin o cos) son muy útiles para modificar la distribución de los datos.  
- En redes neuronales, donde la activación y la propagación de gradientes pueden verse afectadas por entradas con rangos muy amplios o distribuciones sesgadas, aplicar estas transformaciones puede ayudar a estabilizar el entrenamiento.
- Además, cuando se trabaja con datos de recuento o variables que abarcan varios órdenes de magnitud, la transformación logarítmica reduce la asimetría y permite que la red capture de manera más efectiva las relaciones subyacentes.

#### 5. Aplicación y adaptación de estas técnicas en redes neuronales

Aunque el ejemplo utiliza modelos como Ridge y RandomForest para ilustrar el impacto de las transformaciones de características, los mismos principios se aplican en redes neuronales:

- **Escalado y normalización:**  
  Es casi una regla de oro en redes neuronales reescalar las características (por ejemplo, utilizando MinMaxScaler o StandardScaler) para garantizar que los gradientes no se vuelvan demasiado pequeños o grandes. Esto mejora la estabilidad y velocidad de convergencia durante el entrenamiento.

- **Expansión del espacio de características (interacciones y polinomios):**  
  Mientras que las redes neuronales profundas tienen la capacidad de aprender interacciones y relaciones no lineales de manera implícita, en redes superficiales o en casos con pocos datos, incorporar de forma explícita interacciones puede ayudar. Por ejemplo, al incluir características polinomiales se proporciona al modelo información adicional que puede facilitar la aproximación de funciones complejas sin requerir un número excesivo de neuronas o capas.

- **Transformaciones no lineales univariadas:**  
  La aplicación de funciones como log, exp, sin y cos es especialmente importante en redes neuronales cuando se trabaja con datos que no siguen una distribución normal. Estas transformaciones pueden ayudar a homogenizar la escala de las entradas, lo que se traduce en una propagación más eficiente de los gradientes y en una mayor estabilidad durante el entrenamiento.

- **Regularización y control de la dimensionalidad:**  
  Al expandir el espacio de características, se corre el riesgo de aumentar la dimensionalidad de manera excesiva, lo que podría llevar a problemas de sobreajuste. En redes neuronales, esto se compensa mediante técnicas de regularización (como dropout, L2 regularization, etc.) y, en ocasiones, mediante la reducción de dimensionalidad antes de alimentar la red.

- **Interpretabilidad y preprocesamiento manual:**  
  Si bien las redes neuronales modernas pueden aprender representaciones complejas de manera automática, en ciertos escenarios es ventajoso aplicar ingeniería de características para mejorar la interpretabilidad del modelo o para incorporar conocimiento previo del dominio. Esto es especialmente relevante en aplicaciones críticas o en entornos donde la explicabilidad es un requisito.
