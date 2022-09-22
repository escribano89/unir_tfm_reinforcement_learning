# Aprendizaje por refuerzo aplicado a un entorno simulado de asistencia robótica
Repositorio para el contenido relativo al trabajo de fin de máster desarrollado en el Máster de Inteligencia Artificial de la Universidad Internacional de La Rioja [(UNIR)](https://www.unir.net/ingenieria/master-inteligencia-artificial/). (2022)

# Resumen

El aprendizaje por refuerzo es un subconjunto del aprendizaje automático que consiguió un hito muy importante en 2017 cuando Google Deep Mind desarrolló un programa basado en aprendizaje por refuerzo que venció a un jugador profesional del juego de mesa Go ( https://www.deepmind.com/research/highlighted-research/alphago ). Esta forma de aprender mediante la interacción con el entorno constituye un punto de entrada muy interesante para desarrollar robots que puedan asistir a personas con movilidad reducida, de forma que, en el futuro, mejoren la calidad de vida. Por este propósito, se desarrolla un estudio sobre un entorno de asistencia robótica asistida virtual ( https://arxiv.org/abs/1910.04700 ) mediante la aplicación de dos algoritmos de aprendizaje por refuerzo profundo. Estos ejecutan pequeñas tareas necesarias para una persona con problemas de movilidad, tales como rascar una picazón, limpiar un brazo de una persona tumbada o dar de comer, siempre respetando las preferencias de un humano a la hora recibir tal asistencia.

# Estructura

- **PPO**: Implementación del algoritmo de PPO (ver [#Referencias](https://github.com/escribano89/unir_tfm_reinforcement_learning/blob/main/README.md#referencias))
- **TD3**: Implementación del algoritmo de TD3 (ver [#Referencias](https://github.com/escribano89/unir_tfm_reinforcement_learning/blob/main/README.md#referencias))
- Archivo **plotting.ipynb**: Archivo de [Google Collab](https://colab.research.google.com/) empleado para generar las gráficas. Este require de adaptación en base a los archivos que se van a comparar. Hay que cambiar el nombre del csv empleado y modificar los label a mostrar en la gráfica.
- Carpeta **Data**: En esta carpeta esta la información de los experimentos más importantes llevados a cabo en el proyecto. Se detalla en la siguiente sección.
- **requirements.txt**: Dependencias instaladas durante el desarrollo del TFM.

# Carpeta de Data

Esta carpeta está estructurada en 3 subcarpetas principales:

- **trainings**: Esta carpeta contiene los entrenamientos y simulaciones mas relevantes llevados a cabo, tanto en TD3 como en PPO y las gráficas empleadas en la memoria. Esta carpeta se organiza en varios niveles: Algoritmo > Entorno > Robot. Para cada combinación se ha incluido la traza del entrenamiento en csv (training_steps*.csv), los resultados de la simulación (results.csv), los modelos generados para el actor y el crítico (models) y una serie de simulaciones destacadas obtenidas tras aplicar el modelo en el entorno (animated_pngs).
- **videos**: En esta carpeta hay varios videos que muestran algunas de las simulaciones más exitosas para cada tarea.
- **verification_itch**: En esta carpeta están los datos relativos a la verificación de la implementación de los dos algoritmos seleccionados, así como las baselines empleadas.

# ¿Cómo reproducir un experimento?

Para poder reproducir un experimento dado, hay que seguir los siguientes pasos:

1. Primero, hay que seleccionar el algoritmo deseado (carpeta TD3 o PPO) y modificar el archivo de *config.py* para incluir los hiperparámetros deseados y el entorno objetivo.
2. Después se ejecuta el archivo *train.py*. Este muestra una traza de la evolución del entrenamiento. Al finalizar, se generan los modelos para el actor (fichero actor) y el crítico (fichero critic) y se exporta un archivo (*training_steps.csv*) con la evolución del entrenamiento.
3. En este punto, se puede verificar la evolución del entrenamiento mediante el archivo *plotting.py* de la carpeta principal y ajustándolo para adecuarlo a la comparativa que se desee realizar.
4. Para ejecutar una simulación en el entorno en el que se ha entrenado el agente (robot), se ejecuta el archivo *inference.py* que procederá a ejecutar las 100 simulaciones.
5. Al terminar la simulación, se genera una carpeta *rollout* en la que se incluyen las simulaciones más destacadas así como los resultados obtenidos en las 100 iteraciones.

En la carpeta de **data** se encuentran los experimentos realizados, por lo que se pueden usar los archivos generados para replicar cualquier punto de los descritos con anterioridad.

# Videos de Ejemplo

- [Bed Bathing](https://www.youtube.com/watch?v=vXMuxoFqH5g)
- [Feeding](https://www.youtube.com/watch?v=Y1Fe0Z5vt5A)
- [Itch I](https://www.youtube.com/watch?v=QmF8oj_QJhU)
- [Itch II](https://www.youtube.com/watch?v=Q6aMOGWsroE&feature=youtu.be)

# Referencias

- **Artículo de referencia principal**: https://arxiv.org/abs/1910.04700 (Assistive Gym: A Physics Simulation Framework for Assistive Robotics)
- Github: https://github.com/Healthcare-Robotics/assistive-gym
- **Phil Tabor**: Gracias a Phil Tabor por su excelente labor creando la plataforma NeuralNet.AI (https://www.neuralnet.ai/) en la cual se pueden aprender conceptos avanzados de una forma sencilla. Totalmente recomendable. En concreto, su implementación de PPO me ha servido de gran ayuda para guiarme en la compresión e implementación de este (https://github.com/philtabor/Advanced-Actor-Critic-Methods/tree/main/PPO/single/continuous). También tiene una sección muy útil dedicada a DDPG y TD3.
- **Frogames**: Esta plataforma creada por Juan Gabriel Gomila (https://frogames.es/) es un recurso excelente para aprender acerca de diversos temas de Inteligencia Artificial, Ciencia de Datos e incluso Matemáticas. En concreto, tiene una sección donde se profundiza en la compresión e implementación de TD3 que me ha sido de gran utilidad.
- **Lazy Programmer**: En esta web (https://lazyprogrammer.me/) se ofrece el acceso a cursos de pago de gran calidad sobre diversos temas del Machine Learning, Deep Learning y Reinforcement Learning. En concreto estos últimos han sido de gran utilidad para el desarrollo del trabajo de fin de máster.
- **David Silver**: El curso de Reinforcement Learning de David Silver constituye un punto excelente de partida para adquirir una buena base en la comprensión del aprendizaje por refuerzo. (https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
- **Udacity Deep Reinforcement Learning Nanodegree**: En este nanodegree de Udacity (https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) se explican los fundamentos y se ponen en marcha varios proyectos que permiten comprender conceptos complejos desde una perspectiva más práctica.
- **Introducción al Aprendizaje por Refuerzo**: Este libro de Torres (https://torres.ai/aprendizaje-por-refuerzo/) es uno de los mejores recursos escritos en castellano para aprender los fundamentos del Aprendizaje por Refuerzo.

# ¿Dudas? ¿Problemas?

Si tienes alguna duda o has encontrado algún problema, por favor crea una incidencia en la sección de Issues. Gracias! :)
