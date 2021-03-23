# Serving Camembert

Le fichier `serving.ipynb` contient une implémentation permettant d'illustrer l'usage de Ray Serve pour le serving d'un modèle camembert.

## Utilisation en production

Il s'agira pour servir Camembert de faire tourner sur un cloud ou en serveur local le code présent dans le notebook en tant que fichier notebook,
c'est à dire le code suivant : 
```python
#Importing Librairies
from transformers import CamembertForSequenceClassification,CamembertTokenizer, Trainer
import ray
from ray import serve
import requests
import argparse
import torch
import numpy as np


args=argparse.Namespace()
use_gpu = torch.cuda.is_available()
#Use this line if you want ot use a GPU (if available)
# args.device = torch.device("cuda" if use_gpu else "cpu")
#Use this one to use the CPU
args.device = torch.device("cpu")

client=serve.start()


class predict_class:
    def __init__(self,args):
        self.args=args
        self.model=CamembertForSequenceClassification.from_pretrained("/home/amaury/Documents/project_a1/camembert-v1")
        self.tokenizer=CamembertTokenizer.from_pretrained("camembert-base")

        trainer=Trainer(
            model=self.model
        )
        self.trainer=trainer
        self.model.to(args.device)

    def __call__(self,request):
        input=await request.body()
        text=input.decode("utf-8")
        
        tokenized=self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")

        result=self.model(**tokenized)
        class_input=np.argmax(result.logits.data.numpy())
        return({"class":str(class_input)})
        
       
# Use this line if there is a GPU
# client.create_backend("classpredict", predict_class, args, ray_actor_options={"num_gpus": 1})
# Use this one if no GPU
client.create_backend("classpredict", predict_class, args)
client.create_endpoint("classpredict",backend="classpredict", route="/classpredict",methods=["GET","POST"])

```

Il conviendra de modifier le chemin ou se trouve le modèle camembert (téléchargé depuis le notebook d'entraînement).

Cela va créer une API que l'on pourra alors appeler depuis le site web.
