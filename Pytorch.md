### Pytorch

PyTorch es un marco de aprendizaje profundo de código abierto desarrollado principalmente por Facebook AI Research (FAIR). Su popularidad se ha incrementado notablemente en la comunidad científica y en la industria debido a su flexibilidad, a la facilidad para crear prototipos de modelos y a la potencia de sus capacidades de ejecución en GPU. 

PyTorch se sustenta en una API de Python que es sencilla de usar y expresiva. Además, cuenta con un backend en C++ que aprovecha bibliotecas de alto rendimiento como CUDA y cuDNN para acelerar los cálculos en GPU. Esta arquitectura mixta combina la facilidad de prototipado y la familiaridad de Python con la velocidad y eficiencia necesarias para entrenar redes neuronales de gran tamaño. Numerosos equipos de investigación han adoptado PyTorch debido a la forma en que gestiona la construcción y ejecución del grafo computacional, así como a su dinámica en el cálculo de gradientes automáticos.

**Grafo computacional dinámico**  
Uno de los aspectos característicos de PyTorch es el uso de un grafo computacional dinámico. Este grafo se construye en tiempo de ejecución conforme se van realizando las operaciones sobre los datos (tensores). El framework registra los nodos y las conexiones en el grafo a medida que se ejecutan las operaciones aritméticas. La principal ventaja de esta aproximación es la flexibilidad para definir y modificar la arquitectura de las redes neuronales de forma interactiva. En un entorno de investigación o de desarrollo donde se experimenta con múltiples configuraciones, la capacidad de variar la estructura de la red sin recompilar un grafo estático ofrece un flujo de trabajo más ágil.  
Si se necesita cambiar el número de neuronas, añadir ramificaciones o aplicar bucles condicionales para arquitecturas específicas, PyTorch evalúa estos cambios sobre la marcha y actualiza internamente la información del grafo. Este dinamismo contrasta con frameworks que emplean grafos estáticos, donde primero se define la arquitectura y luego se "compila" para su ejecución.

**Visualización de grafos**  
El grafo en PyTorch se construye de modo implícito mientras se efectúa el forward pass de los tensores. Aunque el framework no trae incorporada una herramienta nativa y oficial de visualización de grafos (como la que podría encontrarse en otros entornos), existen librerías de terceros y extensiones que permiten dibujar la estructura de las operaciones registradas. Algunas soluciones se apoyan en bibliotecas como Graphviz o dot para generar diagramas que muestren cómo se conectan las operaciones y los tensores. También se pueden usar herramientas como TensorBoard, que si bien pertenecen originalmente a otros ecosistemas, cuentan con funciones para traducir la secuencia de operaciones de PyTorch a un grafo que se despliega en un panel de visualización. De esta forma, se facilita la inspección y depuración de arquitecturas complejas.

**Tensores y propiedades avanzadas**  
El tensor es la estructura fundamental para almacenar y manipular datos en PyTorch. Un tensor puede tener diferentes dimensiones, desde un escalar (0D) hasta arreglos de alta dimensionalidad (nD). A nivel operativo, los tensores en PyTorch se comportan de manera muy similar a los arrays de NumPy, lo que facilita la transición para quienes están habituados a la biblioteca de Python para cálculos científicos. Sin embargo, a diferencia de NumPy, los tensores de PyTorch permiten la ejecución en GPU sin cambiar el código, simplemente trasladando el tensor a la memoria de la GPU (por ejemplo, con la instrucción `.cuda()` o `.to(device)`).

En PyTorch se pueden crear tensores de diversas formas:  
- A partir de listas o arrays de NumPy mediante `torch.tensor(...)`.  
- Iniciándolos con valores aleatorios, como `torch.rand`, `torch.randn`, `torch.randint`.  
- Generando tensores vacíos (`torch.empty`).  
- Copiando el tamaño o contenido de otros tensores.  

Los tensores en PyTorch tienen propiedades avanzadas como la posibilidad de elegir el tipo de dato (`float32`, `float64`, `int64`, etc.), la opción de requerir gradientes, y métodos para modificar su forma o dimensiones sin cambiar los datos subyacentes. Resulta posible realizar también operaciones in-place (por ejemplo, `tensor.add_(otro_tensor)`) con el cuidado de no romper el seguimiento de gradientes para autograd cuando la operación in-place afecta a valores necesarios en el backward pass.

**Técnicas de slicing y reshape**  
El slicing y el reshape son operaciones cruciales para la manipulación de tensores. En PyTorch, se pueden usar índices regulares, slices (`:`), índices booleanos y máscaras para seleccionar subconjuntos específicos de datos. Por ejemplo:  
```python
x = torch.randn(4, 5)
fila_especifica = x[2]         # Selecciona la tercera fila
sub_tensor = x[:, 1:3]         # Selecciona columnas 1 y 2 de todas las filas
mascara = x > 0                # Tensor booleano que indica dónde x es positivo
valores_positivos = x[mascara] # Extrae todos los valores positivos
```
El reshape se logra con métodos como `view`, `reshape` o `unsqueeze`. Estas funciones permiten reorganizar la forma del tensor sin alterar los datos en sí. Por ejemplo, cambiar de una dimensión `[batch_size, canales, alto, ancho]` a `[batch_size, alto, ancho, canales]` puede ser necesario para ciertas operaciones. La diferencia entre `view` y `reshape` radica en la forma en que PyTorch garantiza la contigüidad de la memoria. `reshape` es más flexible y puede reordenar la memoria de ser preciso, mientras que `view` requiere que el tensor sea contiguo en memoria.

**Autograd**  
Autograd es el sistema de diferenciación automática de PyTorch que registra las operaciones sobre los tensores marcados con `requires_grad=True`. Cuando se ejecuta una operación (suma, multiplicación, convolución, etc.), PyTorch crea un nodo en el grafo que mapea la relación entre los tensores de entrada y los de salida. Este proceso construye de manera dinámica un grafo que describe la ruta computacional de la función.  
Al invocar `tensor.backward()`, PyTorch recorre el grafo en sentido inverso y calcula los gradientes de cada tensor involucrado, propagando estos valores de manera sucesiva hasta llegar a los parámetros iniciales. De este modo, los desarrolladores pueden concentrarse en la implementación del forward pass sin tener que programar explícitamente la derivada de cada operación. Gracias al autograd, se simplifica la tarea de entrenar redes neuronales, puesto que el ajuste de parámetros a través de métodos de optimización se basa en los gradientes calculados automáticamente.

**Extensibilidad**  
PyTorch ofrece múltiples vías para extender el framework. En el nivel básico, la clase `nn.Module` permite crear nuevos componentes o capas customizadas, definiendo un método `forward` que describe la transformación de entrada a salida. Durante la ejecución, autograd sigue automáticamente las operaciones. Para quienes necesitan mayor control, existen funciones de bajo nivel en C++ que pueden incorporarse a PyTorch como extensiones. Esto es útil si se requiere implementar operaciones específicas que no vienen incluidas de forma nativa, o si se buscan optimizaciones en componentes muy particulares.

Además, PyTorch cuenta con subbibliotecas oficiales que extienden sus capacidades en dominios específicos:  
- **TorchVision**: para visión por computador (contiene transformaciones de datos, modelos preentrenados y estructuras para redes convolucionales).  
- **TorchText**: para el procesamiento de lenguaje natural (incluye herramientas para el manejo de vocabularios, tokenización y creación de lotes de secuencias).  
- **TorchAudio**: para la manipulación de datos de audio (transformaciones, carga de datos y modelos enfocados en señales de audio).  

Esta extensibilidad promueve la innovación y la colaboración a través de la comunidad, ya que diferentes grupos pueden desarrollar y compartir módulos o técnicas que simplifiquen las tareas de construcción y entrenamiento de modelos.

**Algoritmos de backpropagation**  
PyTorch implementa la retropropagación de manera automática, calculando la derivada de la función de pérdida con respecto a los parámetros del modelo. Este proceso se conoce como backpropagation e implica aplicar la regla de la cadena para distribuir gradientes a lo largo del grafo computacional. Para redes neuronales recurrentes, se aplica la técnica de Backpropagation Through Time (BPTT), que requiere desenrollar las conexiones recurrentes en un número determinado de pasos para luego propagar gradientes a través de los estados ocultos en cada paso temporal. PyTorch hace esto de forma nativa al registrar las operaciones durante cada iteración en el tiempo.

El método `loss.backward()` desencadena la propagación hacia atrás de los gradientes; una vez que los gradientes se han calculado, se utilizan para actualizar los parámetros del modelo mediante un optimizador apropiado (por ejemplo, SGD, Adam o RMSProp). Las ecuaciones de actualización varían dependiendo del optimizador elegido, pero todas se basan en los gradientes calculados por autograd.

**Máscaras**  
En redes neuronales y en el procesamiento de datos, las máscaras son útiles para seleccionar elementos de un tensor o anular ciertas posiciones durante el entrenamiento o la inferencia. PyTorch facilita la creación de máscaras, ya que los tensores booleanos pueden emplearse para indexar otros tensores y aislar subconjuntos específicos de datos. También se usan máscaras en operaciones como la atención de secuencias (en transformers, por ejemplo), para ignorar posiciones que no deben contribuir al resultado. La función `masked_fill_` o la multiplicación directa por una máscara booleana o de tipo flotante (0s y 1s) son técnicas comunes para su uso.

**Position-wise feed-forward networks**  
Las position-wise feed-forward networks son bloques de transformación frecuentemente utilizados en arquitecturas de modelos de secuencia, especialmente en transformadores (Transformers). Estas redes se aplican de manera independiente a cada posición de la secuencia, combinando capas lineales y funciones de activación como ReLU o GELU. En PyTorch, se implementan con módulos `nn.Linear` y las activaciones correspondientes. La gran mayoría de las implementaciones de transformers en PyTorch (ya sean bibliotecas propias o de terceros como fairseq, Hugging Face Transformers, etc.) utilizan este patrón con un bloque feed-forward de dos capas lineales y una función de activación en medio.  
El carácter position-wise proviene de que cada posición de la secuencia se procesa de forma separada, sin mezclar información entre las posiciones al interior de este sub-bloque. Esto contrasta con la parte de auto-atención, que sí combina información entre posiciones.

**Batches**  
El entrenamiento de redes neuronales a menudo se realiza en lotes (batches) para aprovechar la eficiencia computacional y estabilizar el proceso de optimización. En PyTorch, el primer índice de un tensor suele reservarse para el batch size, mientras que el resto de dimensiones se utilizan para canales, alto, ancho en imágenes, o la longitud de la secuencia en texto, etc. Para manejar estos lotes de datos, PyTorch provee estructuras como `Dataset` y `DataLoader` dentro de `torch.utils.data`, que permiten iterar fácilmente sobre conjuntos de datos grandes, generando lotes de manera automática y habilitando técnicas de muestreo aleatorio.  
La idea básica es definir un `Dataset` que explique cómo se accede a cada muestra (entrada y etiqueta) y, posteriormente, un `DataLoader` que agrupa esas muestras en lotes de un tamaño especificado. Este proceso es esencial para escalabilidad y para aprovechar al máximo la ejecución en GPU, ya que la computación vectorizada de un batch completo es mucho más eficiente que procesar una muestra a la vez.

**Optimizadores**  
PyTorch pone a disposición el módulo `torch.optim` con varios optimizadores de uso común en aprendizaje profundo:  
- **SGD (Stochastic Gradient Descent)**: realiza la actualización del gradiente básico con o sin momentum, y puede incluir decaimiento de peso (weight decay).  
- **Adam**: combina ideas de RMSProp y AdaGrad, ajustando dinámicamente la tasa de aprendizaje para cada parámetro y funcionando bien en muchos problemas.  
- **RMSProp**: mantiene la media cuadrática de los gradientes y normaliza de acuerdo con este historial, útil en problemas con gradientes de distinta magnitud.  
- **Adagrad**: incrementa la tasa de aprendizaje de los parámetros que rara vez se actualizan y la reduce para aquellos más actualizados.  
- **AdamW**, **Adadelta**, **Adamax**, **ASGD**, etc., son variantes o algoritmos complementarios adecuados a distintos escenarios.  

La API de PyTorch para optimizadores requiere proveer los parámetros del modelo (típicamente `model.parameters()`) y configurar hiperparámetros como la tasa de aprendizaje (`lr`), momentum, beta o regularización. Durante el ciclo de entrenamiento, se ejecutan los pasos típicos:  
1. Se realiza el forward pass y se obtiene la salida de la red.  
2. Se calcula la pérdida comparando la salida con el objetivo.  
3. Se ejecuta `loss.backward()` para propagar gradientes.  
4. Se llama a `optimizer.step()` para actualizar los parámetros con base en los gradientes calculados.  
5. Se invoca `optimizer.zero_grad()` para reiniciar los gradientes del modelo antes de la siguiente iteración.

**Label smoothing**  
Label Smoothing es una técnica usada habitualmente en problemas de clasificación, especialmente en el entrenamiento de grandes modelos de lenguaje o de visión por computadora. Consiste en evitar que la distribución de probabilidad aprendida se vuelva demasiado “determinista”. En lugar de usar una vector de etiqueta one-hot estricto (por ejemplo, `[0, 0, 1, 0]`), se introduce una pequeña perturbación que reparte parte de la probabilidad en las clases no verdaderas. De este modo, si la etiqueta real es la clase 2 en un problema de 4 clases, en lugar de tener `[0, 0, 1, 0]`, se podría tener `[0.02, 0.02, 0.94, 0.02]`.  
PyTorch no cuenta con una función de pérdida específica para label smoothing en todas sus versiones, pero se puede implementar usando `nn.CrossEntropyLoss` y transformando las etiquetas, o con el propio `nn.LabelSmoothingLoss` en versiones recientes. El objetivo es mitigar la sobreconfianza del modelo y, en algunos casos, mejorar la generalización.

**Batch normalization**  
La normalización por lotes (BatchNorm) es una técnica que acelera el entrenamiento y estabiliza la distribución de activaciones. PyTorch ofrece módulos como `nn.BatchNorm1d`, `nn.BatchNorm2d` y `nn.BatchNorm3d` según la dimensionalidad de los datos (1D para secuencias, 2D para imágenes, etc.). La idea principal es normalizar la salida de cada canal de manera que tenga media cero y varianza uno dentro de cada batch, aprendiendo además parámetros de compensación y escala para que la red pueda "reajustar" las activaciones en caso necesario.  
Durante el entrenamiento, BatchNorm acumula estadísticas de media y varianza que luego se usan en el modo de evaluación (`modelo.eval()`). Esto ayuda a reducir problemas como el "internal covariate shift" y, en la práctica, tiende a aumentar la tasa de convergencia del entrenamiento.

**Dropout**  
Dropout es otra técnica popular para combatir el sobreajuste. La idea es que, durante el entrenamiento, se "apaguen" (es decir, se pongan en cero) de forma aleatoria un porcentaje de las neuronas. Esto obliga al modelo a no depender demasiado de ciertos activadores específicos y fomenta que el aprendizaje se distribuya en una representación más robusta. En PyTorch, se implementa mediante `nn.Dropout`, `nn.Dropout2d` o `nn.Dropout3d` dependiendo de la dimensionalidad de los datos.  
Al entrenar, Dropout desactiva neuronas al azar con una probabilidad `p`, mientras que, en la fase de evaluación, todas las neuronas permanecen activas, pero se escala su salida para mantener la consistencia estadística. El empleo correcto de dropout puede mejorar la generalización de los modelos, aunque su configuración exacta depende de la arquitectura y la tarea.

**Dropout variacional**  
El dropout variacional surge en el contexto de las redes recurrentes y los modelos bayesianos, donde se busca que la misma máscara de dropout se aplique de manera consistente en todos los pasos temporales para una determinada muestra. Esto implica que las neuronas que se "apaguen" o "enciendan" durante un paso se mantengan de la misma forma en pasos posteriores, generando una regularización coherente a lo largo de la secuencia. Este esquema resulta importante en tareas de modelado de lenguaje o secuencias, pues evita que la red se "olvide" de información significativa entre pasos. PyTorch permite implementar este concepto al manipular directamente las máscaras de dropout o al usar variantes personalizadas de capas recurrentes con dropout.

**Skip connections**  
Las skip connections, o conexiones de salto, se refieren a la práctica de unir la salida de una capa a una capa posterior, saltándose algunas capas intermedias. Son comunes en arquitecturas profundas para mejorar el flujo del gradiente y evitar el problema de la degradación en redes muy grandes. Dichas conexiones permiten que la información se propague más directamente, reduciendo la dificultad de entrenar capas muy profundas.  
En PyTorch, se implementan sencillamente sumando la salida de una capa (o bloque) con la de otra capa más "lejana". Esta acción se puede realizar mediante simples operaciones tensoriales en el método forward de un `nn.Module`. Un ejemplo icónico es la red ResNet, que popularizó la noción de "bloques residuales" con estas conexiones de salto.

**Conexiones residuales**  
Las conexiones residuales (residual connections) son un caso particular de skip connections en el que la salida de un bloque se suma a su entrada. Esto se ve con frecuencia en redes de muy alta profundidad, donde cada bloque está definido por varias capas (normalmente convolucionales, BatchNorm y activaciones), y la entrada del bloque se añade a la salida del mismo. Este mecanismo facilita que el bloque aprenda solo la parte residual que diferencia la entrada y la salida esperada.  
En PyTorch, una implementación típica de un bloque residual combina la entrada `x` con la salida transformada `F(x)` usando la operación `x + F(x)`. En redes como ResNet, DenseNet o en los transformers, este tipo de conexiones se combinan con normalizaciones y capas lineales o convolucionales para componer arquitecturas más complejas y efectivas.

**Implementación práctica de componentes y flujo de trabajo**  
En PyTorch, la estructura esencial de un modelo se define creando una clase que herede de `nn.Module`. Dentro de esta clase, se inicializan las capas requeridas (convoluciones, capas lineales, normalizaciones, etc.) en el constructor (`__init__`), mientras que la secuencia de operaciones del forward pass se implementa en el método `forward`. Así, se puede controlar de manera muy granular cómo fluyen los tensores a través del modelo, lo que facilita la introducción de skip connections, máscara de atención, dropout, etc.  
El entrenamiento involucra iterar sobre un conjunto de datos, dividirlo en batches y, para cada batch, ejecutar un forward pass, computar la pérdida, propagar los gradientes (backward) y actualizar los parámetros del modelo con el optimizador. Diversas rutinas adicionales se implementan para medir desempeño, guardar checkpoints de los pesos o regularizar el entrenamiento.

**Ejecución en GPU**  
Para ejecutar el entrenamiento en GPU, basta con trasladar los tensores y el modelo a la memoria de la GPU mediante sentencias como `model.to(device)` y `tensor.to(device)`, donde `device` suele ser `'cuda'` si se cuenta con una GPU CUDA. Es posible manejar múltiples GPU distribuido en varios enfoques, incluyendo `DataParallel` y `DistributedDataParallel`. Esto permite entrenar modelos muy grandes o procesar enormes volúmenes de datos, siempre que exista suficiente memoria de GPU disponible. PyTorch ha optimizado muchas de sus operaciones para sacar provecho de la paralelización masiva que ofrecen las GPU, reduciendo significativamente los tiempos de entrenamiento respecto a la CPU.

**Uso en arquitecturas diversas**  
PyTorch no se limita a un solo tipo de red. Es posible construir redes fully connected clásicas, redes convolucionales para visión por computadora, redes recurrentes para series temporales y procesamiento del lenguaje, o arquitecturas de auto-atención como los transformers. Las position-wise feed-forward networks y las máscaras desempeñan un papel importante en modelos de atención, facilitando la combinación de señales contextuales a lo largo de secuencias. Las skip connections y residual connections son útiles en la construcción de redes profundas de visión (ResNet, DenseNet) y transformers (donde la normalización y las conexiones residuales se emplean a cada paso de auto-atención y feed-forward).

**Control de hiperparámetros y ajustes**  
En PyTorch, los hiperparámetros, como la tasa de aprendizaje, la tasa de dropout, el número de canales en capas convolucionales o el número de capas en un transformador, suelen definirse como parámetros de inicialización en el modelo o en el optimizador. Los desarrolladores pueden cambiar valores y volver a ejecutar el script de entrenamiento, aprovechando la construcción dinámica del grafo para probar variantes sin mayor dificultad. También se pueden integrar librerías externas para la búsqueda de hiperparámetros (por ejemplo, Optuna, Ray Tune), haciendo que el flujo experimental sea más sistemático.

**Depuración y registro de experimentos**  
Debido a su naturaleza imperativa, PyTorch facilita la depuración de errores. Es posible insertar puntos de control (breakpoints) en el código, imprimir valores de tensores y usar herramientas estándar de Python para inspeccionar la ejecución en tiempo real. Esto contrasta con modelos de grafo estático, donde gran parte de la lógica está "compilada" en una estructura difícil de inspeccionar paso a paso.  
Para registrar experimentos y métricas, se suele recurrir a herramientas como TensorBoard o Weights & Biases, que se integran sin inconvenientes con PyTorch, recogiendo en cada iteración o época las curvas de pérdida, precisión u otras métricas relevantes. Asimismo, se pueden visualizar histogramas de activaciones o gradientes para evaluar si hay saturación o explosión de valores.

**Uso de contenedores y despliegue**  
En escenarios de producción, es frecuente encapsular los entornos de PyTorch en contenedores Docker para asegurar la reproducibilidad y el aislamiento de dependencias. PyTorch también provee un modo de "trazado" llamado TorchScript que convierte un modelo a un grafo estático optimizado, permitiendo su ejecución en entornos sin un intérprete de Python. Esto se vuelve útil para ejecutar inferencias en dispositivos móviles o entornos de baja latencia.

**Integración de todos los componentes**  
PyTorch facilita un ciclo de desarrollo en el que se definen los tensores y la arquitectura del modelo, se aplican técnicas de slicing y reshape para adecuar los datos, se aprovecha el autograd para la retropropagación, se incorporan capas con dropout, normalización por lotes o skip connections, y se entrena usando distintos algoritmos de optimización. Es posible usar posiciones feed-forward networks en transformers y en secuencias, aplicar máscaras para omitir partes de la entrada, realizar label smoothing para regular la clasificación y aprovechar el dropout variacional en redes recurrentes. Cada uno de estos elementos se integra sin fricción debido a la forma modular en que PyTorch expone sus clases y funciones.  

El resultado de esta integración es una plataforma completa que se adapta tanto a la experimentación de vanguardia en investigación de redes neuronales como al desarrollo de soluciones industriales que demandan eficiencia y escalabilidad. En un ecosistema en continuo crecimiento, PyTorch se perfila como una herramienta versátil, apoyada por una gran comunidad de desarrolladores y científicos de datos, que a su vez genera nuevos aportes en forma de paquetes adicionales y ejemplos prácticos.
