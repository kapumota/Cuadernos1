### Artículo importantes

**Redes Generativas Adversarias**

A continuación se presenta el resumen del artículo original sobre las Generative Adversarial Networks. Al leer este resumen, notarás muchos términos y conceptos con los que quizás no estés familiarizado. 

Fuente: https://arxiv.org/abs/1406.2661

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio

Proponemos un nuevo marco para estimar modelos generativos a través de un proceso adversarial, en el cual entrenamos simultáneamente dos modelos: un modelo generativo G que captura la distribución de los datos, y un modelo discriminativo D que estima la probabilidad de que una muestra provenga de los datos de entrenamiento en lugar de G. El procedimiento de entrenamiento para G es maximizar la probabilidad de que D cometa un error. Este marco corresponde a un juego minimax de dos jugadores. En el espacio de funciones arbitrarias G y D, existe una solución única, con G recuperando la distribución de los datos de entrenamiento y D igual a 1/2 en todas partes. En el caso donde G y D están definidos por perceptrones multicapa, todo el sistema puede ser entrenado con retropropagación. No se necesita ninguna cadena de Markov ni redes de inferencia aproximada desenvueltas durante el entrenamiento o la generación de muestras. Los experimentos demuestran el potencial del marco a través de una evaluación cualitativa y cuantitativa de las muestras generadas.

**Attention Is All You Need**


Fuente: https://arxiv.org/abs/1706.03762

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin

Los modelos de transducción de secuencias dominantes se basan en redes neuronales recurrentes o convolucionales complejas en una configuración de codificador-decodificador. Los modelos de mejor rendimiento también conectan el codificador y el decodificador a través de un mecanismo de atención. Proponemos una nueva arquitectura de red simple, el Transformer, basada únicamente en mecanismos de atención, eliminando por completo la recurrencia y las convoluciones. Experimentos en dos tareas de traducción automática muestran que estos modelos son superiores en calidad, al tiempo que son más paralelizables y requieren significativamente menos tiempo de entrenamiento. Nuestro modelo logra 28.4 BLEU en la tarea de traducción de inglés a alemán del WMT 2014, mejorando sobre los mejores resultados existentes, incluidas las combinaciones, en más de 2 BLEU. En la tarea de traducción de inglés a francés del WMT 2014, nuestro modelo establece una nueva puntuación BLEU de estado del arte para un solo modelo de 41.8 después de entrenar durante 3.5 días en ocho GPUs, una fracción pequeña de los costos de entrenamiento de los mejores modelos de la literatura. Demostramos que el Transformer se generaliza bien a otras tareas aplicándolo con éxito al análisis sintáctico del inglés tanto con datos de entrenamiento grandes como limitados.

**GPT-4 Technical Report (Abstract)**


Fuente: https://arxiv.org/abs/2303.08774

Informamos sobre el desarrollo de GPT-4, un modelo multimodal a gran escala que puede aceptar entradas de texto e imagen y producir salidas de texto. Aunque es menos capaz que los humanos en muchos escenarios del mundo real, GPT-4 exhibe un rendimiento a nivel humano en varios puntos de referencia profesionales y académicos, incluyendo aprobar un examen simulado de abogacía con una puntuación en el 10% superior de los examinados. GPT-4 es un modelo basado en Transformer preentrenado para predecir el siguiente token en un documento. El proceso de alineación postentrenamiento resulta en un mejor rendimiento en medidas de factualidad y adherencia al comportamiento deseado. Un componente central de este proyecto fue desarrollar infraestructura y métodos de optimización que se comporten de manera predecible en una amplia gama de escalas. Esto nos permitió predecir con precisión algunos aspectos del rendimiento de GPT-4 basándonos en modelos entrenados con no más de 1/1,000 del cómputo de GPT-4.

**Training Language Models to Follow Instructions with Human Feedback**

Fuente: https://arxiv.org/abs/2203.02155

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe

Hacer que los modelos de lenguaje sean más grandes no los hace inherentemente mejores para seguir la intención del usuario. Por ejemplo, los modelos de lenguaje grandes pueden generar salidas que son falsas, tóxicas o simplemente no útiles para el usuario. En otras palabras, estos modelos no están alineados con sus usuarios. En este artículo, mostramos una vía para alinear los modelos de lenguaje con la intención del usuario en una amplia gama de tareas afinándolos con retroalimentación humana. Comenzando con un conjunto de indicaciones escritas por etiquetadores e indicaciones enviadas a través de la API de OpenAI, recopilamos un conjunto de datos de demostraciones de etiquetadores del comportamiento deseado del modelo, que usamos para afinar GPT-3 usando aprendizaje supervisado. Luego recopilamos un conjunto de datos de clasificaciones de salidas del modelo, que usamos para afinar aún más este modelo supervisado utilizando aprendizaje por refuerzo con retroalimentación humana. Llamamos a los modelos resultantes InstructGPT. En evaluaciones humanas de nuestra distribución de indicaciones, las salidas del modelo InstructGPT de 1.3B parámetros son preferidas a las salidas del GPT-3 de 175B parámetros, a pesar de tener 100 veces menos parámetros. Además, los modelos InstructGPT muestran mejoras en veracidad y reducciones en la generación de salidas tóxicas, mientras tienen regresiones mínimas de rendimiento en conjuntos de datos públicos de NLP. Aunque InstructGPT aún comete errores simples, nuestros resultados muestran que afinar con retroalimentación humana es una dirección prometedora para alinear los modelos de lenguaje con la intención humana.

**Denoising Diffusion Probabilistic Models**


Fuente: https://arxiv.org/abs/2006.11239

Jonathan Ho, Ajay Jain, Pieter Abbeel

Presentamos resultados de síntesis de imágenes de alta calidad utilizando modelos probabilísticos de difusión, una clase de modelos de variables latentes inspirados en consideraciones de la termodinámica fuera del equilibrio. Nuestros mejores resultados se obtienen entrenando con una cota variacional ponderada diseñada según una nueva conexión entre modelos probabilísticos de difusión y coincidencia de puntuación de desenfoque con dinámica de Langevin, y nuestros modelos admiten naturalmente un esquema de descompresión con pérdida progresiva que puede interpretarse como una generalización de la decodificación autoregresiva. En el conjunto de datos incondicional CIFAR10, obtenemos una puntuación de Inception de 9.46 y una puntuación FID de 3.17, ambas de vanguardia. En LSUN de 256x256, obtenemos una calidad de muestra similar a ProgressiveGAN.
