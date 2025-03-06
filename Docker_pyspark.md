### Docker y Pyspark

#### 1. Verificar que Docker Desktop esté instalado y funcionando

Antes de comenzar, asegúrate de que Docker Desktop esté instalado y corriendo en tu Windows 11. Puedes abrir una terminal (CMD, PowerShell o Git Bash) y ejecutar:

```bash
docker --version
```

Esto te confirmará que Docker está correctamente instalado.


#### 2. Descargar y ejecutar la imagen de Docker

El comando que utilizarás es:

```bash
docker run -p 8888:8888 -p 4040:4040 -v %cd%:/home/jovyan/work --name spark-jupyter -d jupyter/pyspark-notebook
```

Desglosemos lo que hace cada parte de este comando:

- **`docker run`**: Inicia un nuevo contenedor.
  
- **`-p 8888:8888`**: Mapea el puerto **8888** del contenedor (usado por Jupyter Notebook) al puerto **8888** de tu máquina. Así, podrás acceder a Jupyter a través de `http://localhost:8888`.
  
- **`-p 4040:4040`**: Mapea el puerto **4040** del contenedor (usado por la interfaz web de Spark) al mismo puerto en tu host. Esto te permitirá monitorear la interfaz de Spark.
  
- **`-v %cd%:/home/jovyan/work`**: Monta el directorio actual (donde ejecutas el comando) en el contenedor, en la ruta `/home/jovyan/work`. Esto te permite tener acceso directo a tus archivos desde el contenedor.
  
- **`--name spark-jupyter`**: Asigna el nombre **spark-jupyter** al contenedor, lo que facilita su manejo.
  
- **`-d`**: Ejecuta el contenedor en segundo plano (modo *detached*).
  
- **`jupyter/pyspark-notebook`**: Es la imagen que contiene Jupyter Notebook con PySpark, Apache Spark, SparkML y otras librerías útiles para machine learning.

*Cuando ejecutes este comando, Docker buscará la imagen en tu máquina y, si no la encuentra, la descargará automáticamente desde Docker Hub.*


#### 3. Consultar los registros del contenedor

Una vez que el contenedor esté en ejecución, necesitarás conocer la URL de Jupyter Notebook (normalmente incluye un token de acceso). Para ello, ejecuta:

```bash
docker logs spark-jupyter
```

En la salida verás información similar a:

```
    To access the notebook, open this file in a browser:
        file:///home/jovyan/.local/share/jupyter/runtime/nbserver-6-open.html
    Or copy and paste one of these URLs:
        http://127.0.0.1:8888/?token=...
```

Copia la URL (con el token) y pégala en tu navegador para acceder al entorno de Jupyter.

#### 4. Detener y reiniciar el contenedor

Cuando termines de trabajar o necesites detener el contenedor, utiliza:

```bash
docker stop spark-jupyter
```

Este comando detendrá el contenedor sin borrarlo, permitiéndote conservar la configuración y los archivos de trabajo.

Para volver a iniciarlo, ejecuta:

```bash
docker start spark-jupyter
```

Al iniciarlo, podrás acceder nuevamente al mismo entorno sin tener que reconfigurar nada.

