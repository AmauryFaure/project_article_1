# Projet d'intégration INSPIRE Article 1

Ce projet a été réalise avec l'association Article 1 par Amaury Faure dans le cadre d'un projet d'intégration à l'École Centrale de Lille. 
Le but du projet est de développer un algorithme permettant de repérer des contenus textuels "dangereux". 

## Description du projet : 

Le fichier `Project_A1_Simple.ipynb` contient des implémentations d'algorithme de modération des messages utilisant la librairie [Scikit-Learn](https://scikit-learn.org/stable/index.html). On utilise aussi la méthode de Bag Of Words ainsi que TF-IDF conjugué à une régression logisitique.

Le fichier `Project_A1_CamemBERT.ipynb` contient une implémentation de [CamemBERT](https://camembert-model.fr/), un réseau neuronal basé sur BERT, entrainé pour la classification de séquence. 

Les 3 autres fichiers contiennent des textes utiles pour l'entraînement et le test des algorithmes :
- `AmauryModerationAllMessagesInspireFrom3Aout2020.xlsx`: Contient des échanges de la plateforme INSPIRE.
- `fr_dataset_test.csv`: Contient des tweets en français provenant de [Multilingual and Multi-Aspect Hate Speech Analysis](https://github.com/HKUST-KnowComp/MLMA_hate_speech), où certains tweets utilisé pour l'entraînement ont été retiré
- `selected_tweets.csv`: un fichier contenant des tweets sélectionnés pour l'entraînement. Plus d'informations sur la sélection des tweets [here](https://github.com/AmauryFaure/get_tweets.git).

## Réutilisation des notebooks :

Pour réutiliser ces notebooks il faudra soit les télécharger et les utiliser en local soit les ouvrir directement dans google colab. Concernant le notebook CamemBERT, étant donné le besoin d'un GPU, je conseille de l'ouvrir avec Google Colab dans un premier temps.

Pour faire tourner les notebooks, j'importe les fichiers présent dans ce répertoire via mon Drive. Il faudra donc probalement changer le chemin des fichiers pour correspondre à votre emplacement.

Une fois cela fait chaque notebook doit pouvoir s'éxécuter.
