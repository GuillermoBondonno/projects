Un resumen de las funciones para preparar, analizar y construir un clasificador de texto segun una oracion sea positiva o negativa.

Usamos los clasificadores SVC y Gaussian Naive Bayes siguiendo la recomendacion de https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html. Al final, estimamos la performance de cada uno, y de los dos combinados, viendo que ambos combinados dan un mejor resultado que SVC, y este que GNB. 

De todas formas, este modelo usa features muy basicos y se guia exclusivamente por la aparicion de palabras claves, asociadas con reseñas positivas o negativas.

#####################################################################################################################################################################

A collection of the functions used to prepare, analyze and build a text classifier to determine the positivity or negativity of a sentence

We use SVC and Gaussian Naive Bayes classifiers according to the recommendations here: https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html . At the end, we estimate the performance of these two and both combined. We see that both classifiers combined produce a decent result.

Either way, this is  a very simple model that works mainly with the appearance of certain key words. 
