# Implementacion-de-una-tecnica-de-aprendizaje-maquina-sin-el-uso-de-un-framework
En este repositorio está el código de un modelo de Machine Learning usando el método de gradient descent.
Se busca predecir el porcentaje de alcohol del vino basándose en la cantidad de ceniza en el vino y el acido malico

El dataset usado cuenta con 178 instancias

params: los 3 parametros que buscamos actualizar --> los que multiplican a las x 
x: muestras dentro del dataset
y: salidas esperadas

Al iniciar el programa hará el entrenamiento del modelo
Para despues proceder al testeo del mismo

Al finalizar se graficarán 2 lineas:
- Linea azul: error en el entrenamiento
- Linea naranja: error en el testeo
