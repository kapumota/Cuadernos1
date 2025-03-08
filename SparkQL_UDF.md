### 1. ¿Qué es Spark SQL?

**Spark SQL** es un módulo del framework Apache Spark que permite trabajar con datos estructurados y semiestructurados mediante un lenguaje SQL, así como a través de una API de DataFrame. Su principal ventaja es que combina la flexibilidad de las consultas SQL con el poder del procesamiento distribuido de Spark. Con Spark SQL, los usuarios pueden:

- Ejecutar consultas SQL interactivas sobre grandes conjuntos de datos.
- Integrar código SQL en aplicaciones basadas en Spark.
- Acceder a datos desde diversas fuentes, como archivos CSV, JSON, bases de datos relacionales, etc.
- Optimizar consultas mediante el motor Catalyst, que permite la optimización lógica y física de las mismas.

El uso de Spark SQL resulta especialmente útil cuando se trabaja con grandes volúmenes de datos y se requiere una interfaz familiar para los analistas y científicos de datos, aprovechando la escalabilidad y la capacidad de procesamiento distribuido de Spark.

### 2. Carga de datos en un DataFrame

En este ejemplo se utiliza el conjunto de datos *mtcars*, un dataset clásico que contiene información sobre automóviles, sus características y métricas de rendimiento.

#### 2.1 Carga en un DataFrame de Pandas

Primero, se carga el archivo CSV desde una URL utilizando la función `read_csv` de Pandas:

```python
# Lee el archivo usando la función `read_csv` de Pandas
mtcars = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0225EN-SkillsNetwork/labs/data/mtcars.csv')
```

#### Explicación:
- **Pandas DataFrame**: Una estructura de datos en Pandas que permite almacenar y manipular datos en formato tabular. Es muy útil para exploración inicial, limpieza y transformación de datos antes de procesarlos con Spark.

Una vez cargado el DataFrame de Pandas, se puede previsualizar parte de su contenido:

```python
# Previsualiza algunos registros
mtcars.head()
```

Es posible que el archivo CSV tenga una columna sin nombre (por ejemplo, "Unnamed: 0") que representa el nombre del automóvil. Se procede a renombrarla para mayor claridad:

```python
mtcars.rename(columns={'Unnamed: 0': 'name'}, inplace=True)
```

Esta operación renombra la columna "Unnamed: 0" a "name", haciendo que el DataFrame sea más legible y preparado para las siguientes operaciones.


### 3. Conversión a un DataFrame de Spark

Una vez que los datos se han cargado en un DataFrame de Pandas, el siguiente paso es convertir estos datos a un DataFrame de Spark para aprovechar las capacidades de procesamiento distribuido. Esto se hace usando la función `createDataFrame` de la SparkSession.

### Código de conversión:

```python
sdf = spark.createDataFrame(mtcars)
```

#### Explicación:
- **createDataFrame()**: Esta función toma un DataFrame de Pandas (u otra fuente de datos) y lo convierte en un DataFrame de Spark, lo que permite realizar operaciones de transformación y acción a gran escala utilizando la infraestructura de Spark.

Para verificar que los datos se han convertido correctamente, se puede visualizar el esquema del DataFrame de Spark:

```python
sdf.printSchema()
```

Este comando muestra la estructura y los tipos de datos de cada columna, lo que es fundamental para asegurarse de que las transformaciones y consultas posteriores se realicen de manera adecuada.

### 4. Manipulación de columnas y renombrado

En muchos casos, es necesario modificar el nombre de las columnas para mejorar la legibilidad o para adaptarlas a ciertos estándares. En este ejemplo se renombra la columna **vs** a **versus** usando el método `withColumnRenamed()`.

#### Código para renombrar la columna:

```python
sdf_new = sdf.withColumnRenamed("vs", "versus")
```

#### Explicación:
- **withColumnRenamed()**: Este método devuelve un nuevo DataFrame con el nombre de una columna cambiado. Es importante notar que el DataFrame original (`sdf`) permanece sin cambios, lo que permite trabajar con versiones modificadas sin alterar la fuente original.

Posteriormente, se puede previsualizar el nuevo DataFrame:

```python
sdf_new.head(5)
```

Esta previsualización confirma que la columna ha sido correctamente renombrada a **versus**.

### 5. Creación de una vista de tabla

Uno de los grandes beneficios de Spark SQL es la capacidad de crear vistas temporales que permiten ejecutar consultas SQL directamente sobre un DataFrame. Esto se logra con el método `createTempView()`.

#### Código para crear una vista temporal:

```python
sdf.createTempView("cars")
```

#### Explicación:
- **createTempView()**: Este método crea una vista de tabla temporal en la sesión de Spark. La vista es solamente visible dentro de la sesión actual y permite que se puedan ejecutar consultas SQL sobre ella utilizando `spark.sql()`.

La creación de una vista facilita la transición de un enfoque programático (usando la API de DataFrame) a uno basado en SQL, lo que es particularmente útil para analistas que prefieren trabajar con SQL.

### 6. Ejecución de consultas SQL y agregación de datos

Una vez que se ha creado la vista de tabla, es posible realizar diversas operaciones utilizando el lenguaje SQL. A continuación se muestran varios ejemplos:

#### 6.1 Mostrar todos los registros

```python
# Muestra toda la tabla
spark.sql("SELECT * FROM cars").show()
```

#### Explicación:
- La consulta `SELECT * FROM cars` devuelve todas las columnas y filas del DataFrame convertido en vista.  
- El método `.show()` imprime un número limitado de registros en la consola, permitiendo una visualización rápida de los datos.

#### 6.2 Seleccionar una columna específica

```python
# Muestra una columna específica
spark.sql("SELECT mpg FROM cars").show(5)
```

#### Explicación:
- Aquí se selecciona solamente la columna **mpg** (millas por galón) de la vista, y se limitan los resultados a los primeros 5 registros.
- Este tipo de consulta es útil para centrarse en variables específicas cuando se analizan datos.

#### 6.3 Consulta con condiciones de filtrado

```python
# Consulta básica de filtrado para determinar qué autos tienen un alto kilometraje y bajo número de cilindros
spark.sql("SELECT * FROM cars WHERE mpg > 20 AND cyl < 6").show(5)
```

#### Explicación:
- La cláusula **WHERE** permite filtrar los registros basándose en condiciones específicas.
- En este caso, se seleccionan los autos que tienen un rendimiento mayor a 20 millas por galón y menos de 6 cilindros, lo que podría indicar vehículos eficientes y ligeros.

#### 6.4 Uso de filtros directamente con la API de DataFrame

```python
# Usa el método where para obtener la lista de autos cuyo millaje por galón es menor a 18
sdf.where(sdf['mpg'] < 18).show(3)
```

#### Explicación:
- Aquí se utiliza la funcionalidad nativa de Spark DataFrame para filtrar los datos. El método `where()` se usa de forma similar a la cláusula **WHERE** en SQL.
- Este enfoque permite aprovechar las ventajas de la API de Spark para realizar operaciones sin necesidad de escribir consultas SQL en forma de cadenas de texto.

#### 6.5 Realizar agregaciones y agrupaciones

```python
# Agrega datos y agrupar por cilindros
spark.sql("SELECT count(*), cyl FROM cars GROUP BY cyl").show()
```

#### Explicación:
- Esta consulta agrupa los registros por el número de cilindros (`cyl`) y cuenta cuántos registros existen en cada grupo.
- La función `count(*)` se utiliza para calcular el número de ocurrencias dentro de cada grupo.
- Las agregaciones son fundamentales en el análisis de datos, ya que permiten resumir y obtener estadísticas sobre grandes conjuntos de datos.


### 7. Creación y Uso de UDFs con Pandas

En el procesamiento de grandes volúmenes de datos, es común que se requieran operaciones personalizadas en columnas específicas. Los **UDFs (User Defined Functions)** permiten definir estas operaciones personalizadas. Tradicionalmente, los UDFs en Spark operaban fila por fila, lo cual implicaba una sobrecarga importante en la serialización de datos. Sin embargo, con los **Pandas UDFs** se puede aprovechar Apache Arrow para realizar operaciones a nivel de columna de manera eficiente.

#### 7.1 ¿Qué es un Pandas UDF?

Un **Pandas UDF** es una función definida por el usuario que utiliza las estructuras de datos de Pandas (Series o DataFrames) para operar sobre particiones completas de datos. Esto significa que en lugar de operar en cada fila individualmente, la función recibe una serie completa (o columna) y devuelve otra serie, lo que resulta en una mejora significativa en el rendimiento.

#### 7.2 Ejemplo: Conversión de la columna “wt”

En el ejemplo, se creará un Pandas UDF para convertir la columna **wt** (peso en 1000 libras) a toneladas métricas. La conversión se realiza dividiendo el valor por un factor adecuado. Aunque los detalles exactos de la conversión pueden variar, el objetivo es demostrar cómo aplicar una transformación a nivel de columna utilizando Pandas UDF.

#### Ejemplo de código para crear un Pandas UDF:

```python
from pyspark.sql.functions import pandas_udf, PandasUDFType

# Define la función de conversión utilizando Pandas
@pandas_udf("double", PandasUDFType.SCALAR)
def lb_to_ton(pd_series: pd.Series) -> pd.Series:
    # Suponiendo que 1 tonelada métrica equivale a 2204.62 libras
    # y dado que la columna 'wt' está en 1000 libras, la conversión se realiza así:
    return pd_series * 1000 / 2204.62

# Aplicar el UDF a la columna 'wt' del DataFrame
sdf_converted = sdf.withColumn("wt_ton", lb_to_ton(sdf["wt"]))

# Mostrar los resultados para verificar la conversión
sdf_converted.select("wt", "wt_ton").show(5)
```

#### Explicación:
- **Decorador @pandas_udf**: Permite definir la función como un Pandas UDF, indicando el tipo de retorno (en este caso, "double") y el tipo de UDF (SCALAR, ya que opera sobre cada columna).
- **Función lb_to_ton**: Esta función toma una Serie de Pandas (representando la columna **wt**) y la transforma multiplicando por 1000 (para convertir de miles de libras a libras) y luego dividiendo por 2204.62, que es el factor de conversión aproximado de libras a toneladas métricas.
- **withColumn()**: Se utiliza para crear una nueva columna, **wt_ton**, que contiene los resultados de la conversión.
- **select() y show()**: Se usan para visualizar la columna original y la columna convertida, permitiendo confirmar que la transformación se ha aplicado correctamente.

#### 7.3 Ventajas de los Pandas UDFs

Los Pandas UDFs permiten:

- **Mejor rendimiento**: Al operar sobre columnas enteras en lugar de procesar fila por fila, se reduce la sobrecarga en la serialización y deserialización de datos.
- **Simplicidad y familiaridad**: Los científicos de datos que están acostumbrados a trabajar con Pandas pueden aplicar sus conocimientos sin necesidad de aprender una sintaxis completamente nueva.
- **Integración con Apache Arrow**: Esto permite un intercambio de datos eficiente entre el JVM de Spark y Python, haciendo que la ejecución de UDFs sea significativamente más rápida.

### 8. Ejemplos adicionales

#### 8.1 Filtrado avanzado y selección de columnas

Imaginemos que queremos extraer únicamente las columnas **name**, **mpg** y **cyl** de la vista y además filtrar los registros para obtener solo aquellos autos con un rendimiento (mpg) superior a 15. El código sería el siguiente:

```python
spark.sql("""
    SELECT name, mpg, cyl
    FROM cars
    WHERE mpg > 15
""").show(10)
```

#### Explicación:
- La consulta SQL selecciona tres columnas específicas y filtra los datos mediante la cláusula **WHERE**.
- Se utiliza el formato multilínea (triple comillas) para hacer el código más legible cuando se tienen consultas más largas.

#### 8.2 Ordenamiento y limitación de resultados

Si se desea ordenar los autos por el rendimiento (mpg) en orden descendente y limitar el número de resultados mostrados, se puede hacer lo siguiente:

```python
spark.sql("""
    SELECT name, mpg
    FROM cars
    ORDER BY mpg DESC
    LIMIT 5
""").show()
```

#### Explicación:
- **ORDER BY mpg DESC**: Ordena los resultados de mayor a menor según la columna **mpg**.
- **LIMIT 5**: Restringe la salida a los primeros 5 registros del resultado ordenado.

#### 8.3 Realización de operaciones de agregación con varias funciones

Podemos realizar operaciones de agregación más complejas, como calcular el promedio de millas por galón y el máximo y mínimo de la potencia de los autos, agrupados por el número de cilindros:

```python
spark.sql("""
    SELECT cyl, AVG(mpg) AS avg_mpg, MAX(hp) AS max_hp, MIN(hp) AS min_hp
    FROM cars
    GROUP BY cyl
""").show()
```

#### Explicación:
- **AVG(mpg)**, **MAX(hp)**, **MIN(hp)**: Se utilizan funciones de agregación para calcular estadísticas sobre el rendimiento y la potencia.
- **GROUP BY cyl**: Agrupa los datos por la columna **cyl** para calcular las estadísticas en cada grupo.

#### 8.4 Uso de funciones integradas y transformaciones

Spark SQL y la API de DataFrame proporcionan múltiples funciones integradas para transformar y analizar los datos. Por ejemplo, si queremos calcular una nueva columna que muestre el doble de la potencia de cada automóvil, podemos hacer:

```python
from pyspark.sql.functions import col

sdf_doble_hp = sdf.withColumn("double_hp", col("hp") * 2)
sdf_doble_hp.select("name", "hp", "double_hp").show(5)
```

#### Explicación:
- **col("hp")**: Se usa para referenciar la columna **hp** en el DataFrame.
- **withColumn()**: Permite crear una nueva columna basada en una operación matemática, en este caso, el doble de la potencia.
- **select()**: Se usa para seleccionar las columnas de interés y visualizar el resultado.

#### 8.5 Ejemplo combinado de operaciones de DataFrame

A modo de ejemplo integral, se puede combinar varias transformaciones y consultas en un solo flujo. Supongamos que queremos filtrar los autos con un alto rendimiento, renombrar una columna, calcular una nueva columna con una transformación y finalmente ordenar el resultado. El siguiente código ilustra cómo se pueden encadenar estas operaciones:

```python
# Filtrar autos con mpg > 20, renombrar 'hp' a 'horsepower', calcular una nueva columna que sea 'wt' en toneladas métricas y ordenar por mpg
sdf_filtered = sdf.filter(sdf["mpg"] > 20) \
    .withColumnRenamed("hp", "horsepower") \
    .withColumn("wt_ton", lb_to_ton(sdf["wt"])) \
    .orderBy("mpg", ascending=False)

sdf_filtered.select("name", "mpg", "horsepower", "wt", "wt_ton").show(5)
```

#### Explicación:
- **filter()**: Se usa para filtrar registros que cumplen con la condición de que el rendimiento (mpg) sea mayor a 20.
- **withColumnRenamed()**: Renombra la columna **hp** a **horsepower** para mayor claridad.
- **withColumn()**: Aplica el UDF `lb_to_ton` para convertir el peso a toneladas métricas.
- **orderBy()**: Ordena el DataFrame según la columna **mpg** de manera descendente.
- **select()**: Permite seleccionar y mostrar las columnas clave para validar las transformaciones realizadas.


### 9. Más detalles sobre Pandas UDFs y su funcionamiento interno

#### 9.1 El rol de Apache Arrow

Los **Pandas UDFs** se benefician del uso de Apache Arrow, un framework de interoperabilidad de memoria que permite el intercambio eficiente de datos entre procesos. Gracias a Arrow, los datos se pueden transferir entre Spark (escrito en Scala y Java) y Python sin incurrir en una gran sobrecarga de conversión. Esto se traduce en un mejor rendimiento, especialmente cuando se procesan grandes volúmenes de datos.

#### 9.2 Ejemplo adicional de Pandas UDF: normalización de datos

Imaginemos que queremos normalizar una columna numérica (por ejemplo, **mpg**) para que sus valores se encuentren en un rango de 0 a 1. Podemos definir un Pandas UDF que realice esta operación:

```python
@pandas_udf("double", PandasUDFType.SCALAR)
def normalize_mpg(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min())

# Aplicamos la función de normalización a la columna 'mpg'
sdf_normalized = sdf.withColumn("mpg_normalized", normalize_mpg(sdf["mpg"]))
sdf_normalized.select("name", "mpg", "mpg_normalized").show(5)
```

#### Explicación:
- La función **normalize_mpg** toma una Serie de Pandas, calcula el valor mínimo y máximo y aplica la fórmula de normalización.
- Se utiliza `withColumn()` para agregar la columna **mpg_normalized** al DataFrame.
- Este ejemplo ilustra cómo se pueden definir transformaciones personalizadas de manera eficiente.

#### 9.3 Registro de UDFs para uso en SQL

Los UDFs también pueden registrarse para que sean utilizados directamente en consultas SQL. Esto se hace con el método `spark.udf.register()`. Por ejemplo, para registrar el UDF de conversión de peso:

```python
spark.udf.register("lb_to_ton_udf", lb_to_ton)
spark.sql("""
    SELECT name, wt, lb_to_ton_udf(wt) AS wt_ton
    FROM cars
    LIMIT 5
""").show()
```

#### Explicación:
- **spark.udf.register()**: Registra el UDF bajo el nombre `"lb_to_ton_udf"`, lo que permite invocarlo en consultas SQL.
- En la consulta SQL se usa el UDF registrado para transformar la columna **wt**.



### 10. Ejemplos prácticos y escenarios de uso

A continuación se muestran escenarios adicionales y ejemplos prácticos que pueden surgir en proyectos reales:

#### 10.1 Análisis exploratorio de datos (EDA)

En muchos proyectos, después de cargar y transformar los datos, es necesario realizar un análisis exploratorio. Por ejemplo, se pueden calcular estadísticas descriptivas utilizando consultas SQL:

```python
spark.sql("""
    SELECT
        COUNT(*) AS total_autos,
        AVG(mpg) AS promedio_mpg,
        MIN(mpg) AS mpg_minimo,
        MAX(mpg) AS mpg_maximo
    FROM cars
""").show()
```

Esta consulta ayuda a entender la distribución del rendimiento de los automóviles en el conjunto de datos.

#### 10.2 Unión y combinación de DataFrames

En escenarios reales, puede ser necesario combinar información de diferentes fuentes. Imaginemos que tenemos otro conjunto de datos (por ejemplo, información adicional de autos) y deseamos unirlo con nuestro DataFrame principal. Esto se podría lograr utilizando el método `join()`:

```python
# Supongamos que tenemos un DataFrame adicional con información extra de autos
extra_info = pd.DataFrame({
    'name': ['Mazda RX4', 'Datsun 710', 'Hornet 4 Drive'],
    'origin': ['USA', 'Japan', 'USA']
})
sdf_extra = spark.createDataFrame(extra_info)

# Realizamos una unión (join) entre el DataFrame original y el DataFrame extra basado en la columna 'name'
sdf_joined = sdf.join(sdf_extra, on="name", how="left")
sdf_joined.show(5)
```

#### Explicación:
- Se crea un DataFrame adicional con información de origen.
- Se utiliza el método `join()` para combinar los DataFrames basándose en la columna **name**.
- La unión **left** garantiza que todos los registros del DataFrame original aparezcan, complementados con la información adicional cuando esté disponible.

#### 10.3 Uso de funciones de ventana

Las funciones de ventana son muy útiles para realizar cálculos sobre particiones de datos. Por ejemplo, se puede calcular una media móvil sobre alguna métrica. Aunque el ejemplo en este laboratorio es sencillo, en escenarios más complejos se pueden aplicar funciones de ventana para análisis temporales o agrupaciones avanzadas.

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import avg

# Definimos una ventana basada en la columna 'cyl'
windowSpec = Window.partitionBy("cyl").orderBy("mpg")

# Calculamos la media móvil (acumulada) para 'mpg' dentro de cada grupo de cilindros
sdf_window = sdf.withColumn("avg_mpg", avg("mpg").over(windowSpec))
sdf_window.select("name", "mpg", "cyl", "avg_mpg").show(5)
```

#### Explicación:
- **Window.partitionBy()**: Define cómo se agrupan los datos en la ventana.
- **Window.orderBy()**: Ordena los datos dentro de cada partición para calcular la media móvil.
- **avg().over(windowSpec)**: Aplica la función de promedio sobre la ventana definida.


### Análisis global y buenas prácticas en el código

La estructura del [código presentado](https://github.com/kapumota/Cuadernos1/blob/main/SparkSQL.ipynb) se basa en varios conceptos fundamentales de PySpark y Spark SQL. A continuación, se destacan algunos aspectos clave y buenas prácticas que se han seguido:

#### Uso de UDFs basados en Pandas
- **Ventajas del enfoque vectorizado**:  
  Al utilizar Pandas UDFs, las operaciones se realizan sobre columnas completas en lugar de procesar cada fila individualmente. Esto permite aprovechar la vectorización de Pandas, reduciendo la sobrecarga de procesamiento y mejorando el rendimiento en comparación con UDFs tradicionales.
- **Compatibilidad y facilidad de registro**:  
  Registrar los UDFs con `spark.udf.register()` permite integrarlos directamente en consultas SQL. Esto facilita la colaboración entre equipos de análisis, ya que los usuarios pueden invocar funciones personalizadas mediante una sintaxis SQL familiar sin tener que escribir código Python adicional.

#### Operaciones de unión (JOIN)
- **Consolidación de datos**:  
  Realizar un JOIN entre DataFrames es una operación común cuando se requiere combinar información de diferentes fuentes. En el ejemplo, se unen datos de empleados (con información básica) con datos salariales, utilizando la columna común `emp_id`.
- **Especificación del tipo de JOIN**:  
  Se ha utilizado un `inner join` para garantizar que solo se incluyan las filas donde la clave de unión esté presente en ambos DataFrames. Esta es una práctica habitual para evitar la inclusión de registros incompletos.

#### Manejo de valores nulos
- **Importancia de la limpieza de datos**:  
  Los datos reales a menudo contienen valores faltantes. El uso de `fillna()` es una forma eficiente de asegurar que estos valores nulos sean reemplazados por valores predeterminados, permitiendo que las operaciones subsecuentes (como cálculos o agregaciones) se realicen sin errores.
- **Aplicación selectiva**:  
  Al pasar un diccionario a `fillna()`, se puede especificar qué columnas deben ser rellenadas y con qué valor, proporcionando flexibilidad y control sobre el proceso de limpieza.

#### Aplicación de consultas SQL en Spark
- **Facilidad para realizar filtros y transformaciones**:  
  La integración de Spark SQL permite realizar operaciones complejas, como filtrar datos basados en patrones (usando `LIKE`) o aplicar funciones de transformación definidas por el usuario, de una forma sencilla y familiar para los usuarios con experiencia en SQL.
- **Visualización interactiva**:  
  Métodos como `.show()` permiten a los usuarios inspeccionar los resultados en tiempo real, lo cual es esencial para la depuración y validación de los datos durante el desarrollo de pipelines de datos.

#### Documentación y comentarios en el código
- **Claridad en la intención**:  
  Cada bloque de código incluye comentarios que explican la intención detrás de la operación. Por ejemplo, se explica que el UDF `convert_wt` se utiliza para convertir unidades de peso de imperial a métricas, y se documenta el factor de conversión utilizado.
- **Facilita la colaboración**:  
  Una buena documentación interna es fundamental para que otros desarrolladores o analistas puedan entender y mantener el código, especialmente en proyectos colaborativos o a largo plazo.

### Integración en flujos de trabajo complejos

El código y las técnicas presentadas son representativos de un flujo de trabajo de análisis de datos en PySpark, en el que se integran diversas operaciones:

- **Definición y registro de UDFs personalizados**: Permite extender la funcionalidad de Spark SQL con transformaciones específicas que no se encuentran entre las funciones integradas.
- **Unión de múltiples fuentes de datos**: La capacidad de realizar JOINs entre DataFrames es esencial para consolidar información de distintas fuentes, ya sea para análisis financiero, de recursos humanos u otros escenarios.
- **Manejo de datos faltantes**: Garantiza la integridad de los datos y evita errores en operaciones subsecuentes, siendo una parte crítica en la limpieza y preparación de datos para análisis o modelado.
- **Ejecución de consultas SQL**: La posibilidad de ejecutar consultas SQL sobre vistas temporales en Spark permite a los usuarios utilizar un lenguaje declarativo y familiar para extraer insights de grandes conjuntos de datos.

El uso combinado de estas técnicas no solo optimiza el procesamiento de datos en entornos distribuidos, sino que también facilita la transición desde un entorno local (como un DataFrame de Pandas) a un entorno distribuido utilizando Spark. Esto es particularmente valioso en proyectos de big data, donde la eficiencia y la escalabilidad son esenciales.

### Aspectos técnicos y consideraciones adicionales

#### Conversión de unidades y factores de multiplicación
- En ambos UDFs presentados se utilizan factores de conversión (0.45 para convertir el peso y 0.425 para convertir el rendimiento).  
- Es fundamental asegurarse de que estos factores sean correctos y estén basados en estándares de conversión confiables.  
- En aplicaciones reales, se recomienda incluir comentarios o referencias sobre la procedencia de dichos factores para garantizar la reproducibilidad y la precisión de las conversiones.

#### Ejecución distribuida y procesamiento en paralelo
- El uso de Pandas UDFs permite que la operación se ejecute en paralelo sobre particiones de datos, aprovechando el entorno distribuido de Spark.  
- Esto mejora significativamente el rendimiento cuando se trabaja con grandes volúmenes de datos, ya que la serialización y deserialización de datos se optimizan mediante Apache Arrow (tecnología subyacente en la implementación de Pandas UDFs).

#### Registro y reutilización de UDFs
- Registrar un UDF en Spark con `spark.udf.register()` permite reutilizar la misma función en múltiples consultas SQL sin necesidad de redefinirla.  
- Esto fomenta la modularidad y el mantenimiento del código, ya que cualquier cambio en la lógica de conversión se puede realizar en un solo lugar y se reflejará en todas las consultas que utilicen ese UDF.

#### Manejo de errores y validación de datos
- El ejemplo de rellenado de valores nulos con `fillna()` es una buena práctica para evitar que la presencia de datos incompletos interrumpa el flujo de trabajo.  
- Es importante incluir validaciones y comprobaciones en cada etapa del procesamiento de datos para asegurarse de que el conjunto de datos cumpla con los requisitos esperados antes de aplicar transformaciones o ejecutar consultas complejas.
