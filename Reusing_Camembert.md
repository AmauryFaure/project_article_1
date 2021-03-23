# Ré-utilisation du notebook Camembert :

Le fichier `Project_A1_CamemBERT.ipynb` contient un entraînement du modèle Camembert disponible sur la librairie Huggingface Transformers.

Ce fichier vient expliquer comment le ré-utiliser.
Je conçoit ici des changements seulement dans la partie création du dataset et changement de notation. Cela crée quelques "incohérences" mais permet une réutilisation plus simple du code dans la suite du notebook.



## Réentrainement de Camembert

### Changement des fichiers de test et entrainement

Selon l'utilisation, il faut changer les fichiers utilisés en tant que fichier d'entraînement et de test (dans les portions correspondantes du notebook).
Je recommande de conserver les formats présents pour que le notebook continue de fonctionner.

C'est à dire, le dataset d'entraînement étant une `DataFrame` avec une colonne `content` et une colonne `Harmful` il faut respecter ce format.

Indication : Vous pouvez montrer le contenu d'une dataframe en utilisant `dataframe_name.head()` (cela affiche les 5 premières lignes). 

Pour le dataset de test, à terme il n'y aura pas besoin du dataset de test MLMA, seulement celui d'Inspire, 
en revanche celui d'inspire doit avoir une colonne `tweet` pour le contenu mais aussi une colonne `sentiment` pour indiquer le
statut harmful ou non du contenu (les noms sont modifiés comparé à content/harmful pour simplifier la réutilisation du code par la suite)
afin de pouvoir calculer les indicateurs de performance du modèle CamemBERT. Le dataset de test avec de contenu d'INSPIRE doit avoir le nom `test` et non `df_a1_to_moderate`.


### Ré-entrainement et test de la performance

Si tout a été respecté jusqu'ici, aucune modificacion n'est nécéssaire pour faire tourner la partie `Implémentation de Camembert` du notebook.
Une fois l'entraînement réalisé, c'est à dire une fois que cette cellule à été éxécuté : 
```python
#Utility to load dataset in batch
from torch.utils.data import DataLoader
#Importing camembdert and AdamW Optimizer
from transformers import CamembertForSequenceClassification, AdamW

#Model declaration
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = CamembertForSequenceClassification.from_pretrained('camembert-base')
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

#Training
for epoch in range(1):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels.long())
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()
```
On peut sauvegarder le modèle pour une réutilisation extérieure au notebook. 
Pour cela il faut utiliser la celulle un peu plus bas (entre matrice de confusion et les indicateurs de performances) :
```python
model.save_pretrained("/content/drive/MyDrive/article_1_model/camembert_v1")
```

Il faut la décommenter dans le notebook, et changer le chemin dans lequel on souhaite sauvegarder le modèle.

### Modifications des hyperparamètres :

Pour améliorer les performances du modèle camemBERT, on peut chercher à modifier certains paramètres, notamment :

- `epoch` : Le nombre d'epoch correspond au nombre de fois ou l'on va effectuer une étape forward et backward du réseau neuronal, globalement ça correspond au nombre de fois ou l'on va montrer les inputs d'entrainement au modèle afin qu'il modifie ses poids interne. Un nombre d'epoch petit risque de ne pas modifier assez les poids, un nombre trop grand risque de trop les modifier. Il faut donc trouver le bon équilibre. 
- `batch_size` : C'est le nombre d'inputs que le modèle va voir à chaque fois. 
- `learning_rate` ou `lr` : c'est un coefficient qui vient modifier la capacité d'apprentissage du modèle. Un learning rate important permet de s'adapter plus vite aux inputs montrés, mais peut trop spécialiser l'algorithme.

Pour trouver la bonne combinaison de paramètres, il faut re-éntraîner camemBERT avec différents paramètre et évaluer sa performance sur un set de données de test.
On chosira alors les paramètres donnant la meilleure performance. 
