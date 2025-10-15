# **Texte du cours**

# [Initiez-vous à l'apprentissage semi-supervisé ]

**Pour vous aider, voici quelques bonnes pratiques chez OpenClassrooms :** 

* Utilisez le **vouvoiement singulier** : vous parlez à un étudiant, vous vous adressez directement à lui.

* Pensez **opérationnel** : l’étudiant est là pour acquérir des compétences qu’il pourra réutiliser dans le monde du travail : soyez donc pragmatique et apportez-lui des informations, techniques, procédures concrètes. 

* Restez **simple** : ne pas simplifier à l'extrême, mais faciliter la tâche des apprenants. Visez la clarté. Faites des phrases courtes. Définissez les nouveaux termes. Pensez à la façon dont le cours sera utilisé par l’apprenant, à la fois comme expérience d'apprentissage en direct et comme référence ultérieure. Dans chaque chapitre, divisez votre contenu en sections thématiques. Il est ainsi plus facile pour les apprenants de retenir les connaissances, d'organiser leur prise de notes éventuelle et de se référer au contenu en cas de besoin.

* **Structurez votre chapitre :** Il est souvent utile d'aborder chaque chapitre en commençant par le pourquoi, puis le quoi, puis le comment.   
  En d'autres termes : Nous devrions apprendre immédiatement pourquoi nous devrions terminer ce chapitre.  Quel est le problème que ce chapitre nous aidera à résoudre ? Quelles sont les informations essentielles dont nous avons besoin pour résoudre le problème ? Comment le résoudre ? Quel est le processus de résolution ? Veillez à fournir un exercice pour mettre ce processus en pratique.

* **Utilisez des transitions**, en soulignant la façon dont les compétences s'appuient les unes sur les autres, en particulier au début de chaque chapitre, mais aussi à la fin de chaque partie et à la fin du cours. Que sommes-nous désormais capables de faire et quelle est la prochaine étape à franchir pour continuer à progresser ?

**Dans le corps du texte, vous avez la possibilité de mettre en avant visuellement des encarts avec de la couleur pour  :** 

//bleu clair 3//

**Donner une information** : (définition, apport de contexte, complément d’information, ressources à consulter…

//orange clair 3//

**Apporter un point de vigilance** 

//rouge clair 3//

**Montrer un exemple d’erreur** ou un interdit

//gris clair 2//

**Poser une question** : on se met à la place de l'étudiant : une question qu'il pourrait se poser en lisant “Mais comment on fait cela ?” ; “Mais à quoi cela sert-il ?” … 

Pour les questions posées de votre point de vue d'expert, n'utilisez pas cette mise en forme. Par exemple, vous pouvez demander à l’apprenant de réfléchir à quelque chose avant de poursuivre sa lecture. C'est une excellente façon d'impliquer l’apprenant, mais ce n'est pas la raison d'être de cet encadré gris.

* | *Citer une phrase et/ou donner un exemple.*

**Si vous écrivez un cours sur la programmation…**

* Indiquez le **code in-line** (qui apparaît dans le texte directement) en formattant le texte avec la police `Roboto Mono`.

* Pour indiquer un **extrait de code**, 

  * ajoutez une ligne de trois backquotes avant et après le code, 

  * précisez entre parenthèses, en amont du code, le langage utilisé.

\`\`\`(javascript)

\[le code\]

\`\`\`

| Pour plus d’informations, vous pouvez consulter cette page : [Conseils de rédaction du cours](https://openclassrooms.notion.site/R-daction-du-cours-5bdf2035acdf481480934da87c998853) |
| :---- |

**1500 mots maximum** par chapitre (Pour compter les mots du Google Doc : sélectionner le chapitre concerné puis cliquer sur outil \> nombre de mot ; une fenêtre s’ouvre et affiche le nombre sélectionné sur le total de mot du document. 📊

# **Introduction**

###### #TLK - Teaser 

Les données brutes existent en masse. Or, la plupart de ces données n'est pas labellisée. Comment entraîner une IA dans ces conditions ? 🤔

C’est exactement le défi des data scientists aujourd'hui. Les données ? Abondantes. Mais les annotations sont rares et coûteuses : il faut du temps, de la précision et de l'expertise pour les ajouter. ⏱️💸

C'est là qu'entre en jeu l’**apprentissage semi-supervisé** ou SSL (*Semi Supervised Learning* en anglais). 🚀

Entre le tout-étiqueté du supervisé et le zéro-étiquette du non supervisé, le semi-supervisé utilise les deux : il apprend à partir d’un petit échantillon bien annoté… pour mieux exploiter la masse restante. ⚖️🧠

Résultat ? Moins d’étiquettes, mais des modèles puissants, capables d’analyser des images médicales avec précision. 🎯

Dans ce cours, vous construirez un modèle de segmentation semi-supervisé pour analyser des images radiologiques. 🩻

Envie de plonger au cœur de l’IA, même quand elle n’a pas toutes les réponses ? Commencez le cours dès maintenant ! 🏊‍♀️🤖

 **Prérequis, outils :** 

Pour suivre ce cours, vous devez être à l’aise avec Python et les bases du machine learning (classification, sur‑apprentissage, métriques), et vous aurez besoin de maîtriser ces outils : 
//bleu clair 3//
* [scikit-learn](https://scikit-learn.org/stable/)  
* [TensorFlow](https://www.tensorflow.org/?hl=fr)  
* [PyTorch](https://pytorch.org/)  
* [OpenCV](https://opencv.org/)

Par ailleurs, avant de vous lancer dans l’apprentissage semi-supervisé, il est préférable d’avoir suivi au préalable les cours suivants : 

* [Initiez-vous au Machine Learning](https://openclassrooms.com/fr/courses/8063076-initiez-vous-au-machine-learning)  
* [Maîtrisez l'apprentissage supervisé](https://openclassrooms.com/fr/courses/8431846-maitrisez-lapprentissage-supervise)  
* [Appréhendez les enjeux métier de l’apprentissage supervisé](https://openclassrooms.com/fr/courses/8468461-apprehendez-les-enjeux-metier-de-l-apprentissage-supervise)

\[lister les pré-requis et outils\].

# **Mettez en place un modèle semi-supervisé**

### P1C1 : Tirez un maximum de ce cours

> Email — Emma (PM)
>
> Objet: Kick-off Hackathon MedTech (48h)
>
> Bonjour l'équipe,
>
> Notre défi: livrer un POC d’aide au tri dermatologique en 48h. Beaucoup d’images, peu d’annotations. Objectif: une démo fluide avec des métriques crédibles (AUC/F1) et une narration claire. On priorise des incréments à fort impact, reproductibles et présentables. Go!
>
//bleu clair 3//
**Brief produit (extrait documentation projet)**
- Problème: peu de labels, masse d’images à trier.
- Contraintes: temps, budget, sécurité patient.
- Critères de succès: uplift mesurable + démo stable.
- Approche: SSL (pseudo-labeling, graphes, cohérence, SGAN en option).

//orange clair 3//
**Note clinique — Dr Malik**
- Classes sensibles: erreurs coûteuses (fausses rassurances).
- Exigences: transparence, calibration, pilotage par seuils.

#### Découvrez l’objectif du cours

Dans ce cours, vous allez comprendre quand et pourquoi utiliser l’apprentissage semi-supervisé (SSL) et apprendre à l’implémenter en Python pour des problèmes d’imagerie médicale. À la fin, vous saurez : 😎

- Identifier les situations où les labels manquent et où le SSL apporte un gain concret. 🔍
- Mettre en place un pipeline de pseudo-labeling avec PyTorch. 🧪
- Exploiter la structure des données via la propagation d’étiquettes sur graphe. 🕸️
- Comprendre le principe des SGANs et de la régularisation par cohérence pour stabiliser les prédictions. 🧘‍♂️

#### Rencontrez votre professeur

MEET YOUR TEACHER VIDEO

Expert en analyse de données, Aurélien Quillet a commencé sa carrière en développant des outils de bio-informatique avant de se spécialiser dans l'apprentissage automatique (Machine Learning). Cette évolution lui a permis d'acquérir une solide expérience dans l'utilisation des tests statistiques et des probabilités au travers de divers projets professionnels couvrant différents secteurs (biologie, finance, marketing, etc.). Formateur depuis 2020, Aurélien Quillet est passionné par le partage de ces connaissances essentielles au développement optimal des entreprises.

#### Découvrez le fonctionnement du cours

Connaissez-vous le principe d'un cours en ligne sur OpenClassrooms ? Ce cours suit une progression logique que l'on a séquencée en plusieurs chapitres, qu'il est préférable de suivre dans l'ordre. 📚

Dans ces chapitres, vous trouverez :

* du **texte** avec des explications et des exemples concrets, pour présenter des outils spécifiques et lister des ressources externes à consulter ou encore des fichiers à télécharger ;  
* des **tutoriels vidéos** permettant de suivre étape par étape la réalisation du projet fil rouge directement sur l'écran de l’expert formateur. Les étapes et les lignes de code vous sont présentées grâce à Google Colab. Google Colab est un environnement en ligne gratuit pour exécuter du code Python dans des notebooks Jupyter. Notez que les résultats obtenus dans le texte et les screencasts peuvent légèrement varier à l'exécution des scripts.

Les sections "À vous de jouer" sont l'occasion de mettre en pratique ; c’est là que vous suivrez notamment les démonstrations en vidéo et que vous pourrez reproduire.

À la fin du cours, vous trouverez un **quiz** pour vous permettre de valider ce que vous avez appris. 🧪✅

Certains blocs de code contiennent des **lignes tronquées**. Pas d'inquiétude, il n'y a pas d'erreurs. Si vous cliquez sur le bloc de code en question, l'intégralité de celui-ci apparaîtra correctement. 💡

Avant de démarrer, voici quelques conseils pour exploiter au mieux le contenu de ce cours et **optimiser votre apprentissage** :

1. Lisez le texte dans chaque chapitre pour comprendre **pourquoi** les concepts abordés sont importants.  
2. Suivez les démonstrations pour savoir **comment** vous pouvez mettre en œuvre ces concepts.  
3. Profitez de chaque occasion de pratiquer en faisant une pause dans le cours, pour vous entraîner de votre côté et reproduire pas à pas ce que vous avez appris.

#### Découvrez le projet fil rouge du cours 🧵

Tout au long du cours, vous allez mettre en place des techniques semi-supervisées sur un dataset médical standardisé. Nous utiliserons `DermaMNIST` (famille MedMNIST) pour illustrer la classification avec peu d’images étiquetées et un grand volume non étiqueté. Vous pourrez ensuite transposer ces techniques à un contexte de segmentation radiologique.

#### Téléchargez la fiche résumé du cours 📝

COURSE SUMMARY  

*Prêt à démarrer ? On pose les bases du semi‑supervisé juste après 🧭* 

### P1C2  : Découvrez le principe de l’apprentissage semi supervisé 🧭

//bleu clair 3//
**Note technique (panorama SSL)**
- Pseudo‑labeling: rapide, simple, sensible au seuil.
- Graphes (LabelSpreading): exploite la structure globale.
- Cohérence (Mean Teacher / UDA / FixMatch): stabilité et calibration.
- SGAN: renforce le discriminateur via une classe supplémentaire (K+1).

//gris clair 2//
**Decision log #1**
- Décision: démarrer par pseudo‑labeling pour un gain rapide.
- Prévoir LabelSpreading pour capturer la structure.
- Garder SGAN pour un effet différenciant si timing OK.
- Justification: coût/valeur/risques au format hackathon.

//orange clair 3//
**Alerte**: calibration et fuite d’info. Toujours valider sur un test set tenu à l’écart.

#### Rappelez-vous des principes du supervisé vs non supervisé

Le supervisé apprend à partir de paires `(x, y)` et nécessite beaucoup d’annotations. Le non supervisé apprend des structures sans étiquettes. Le semi-supervisé combine un petit ensemble annoté avec un large ensemble non annoté pour améliorer la généralisation à moindre coût. ⚙️

//orange clair 3//
**Point de vigilance**: les erreurs de pseudo-labels peuvent se propager. Gardez des seuils de confiance et évaluez régulièrement sur un jeu de test labellisé. ⚠️

#### Comprenez l’intérêt de l’apprentissage semi-supervisé

- Réduire le coût d’annotation (experts rares en médical). 💸
- Mieux exploiter la masse de données disponibles en production. 📈
- Améliorer la robustesse et la calibration des modèles quand les labels sont bruyants ou partiels. 🛡️

#### Comprenez les principes fondamentaux du semi-supervisé

- Pseudo-labeling: utiliser les prédictions les plus confiantes comme labels temporaires. 🏷️
- Méthodes par graphe: propager l’information des nœuds labellisés vers leurs voisins dans un espace de similarité. 🕸️
- Régularisation par cohérence: forcer des prédictions stables malgré de petites perturbations (Mean Teacher, UDA). 🔁

#### En résumé 

* Le SSL tire parti des données non étiquetées pour renforcer l’apprentissage.
* Des seuils et une évaluation continue sont indispensables.
* Plusieurs familles de méthodes existent et sont complémentaires.

*Dans le prochain chapitre, vous verrez comment implémenter pas à pas le pseudo-labeling en PyTorch.* 👉

### P1C3 : Catégorisez l’inconnu avec la pseudo labellisation 🧪

//bleu clair 3//
**Playbook — Pseudo‑labeling (documentation projet)**
1. Entraîner un modèle de base sur peu de labels.
2. Prédire sur non étiquetées + softmax.
3. Sélectionner au‑dessus d’un seuil (confiance/écart top‑1 vs top‑2).
4. Ajouter ces exemples à l’entraînement.
5. Réentraîner, surveiller AUC/F1.

> Email — Emma
>
> Point intermédiaire: priorité à un uplift tangible, pas de sur‑promesse. Partagez vos seuils et décisions dans la doc.

//orange clair 3//
**Alerte**: biais de confirmation. Un seuil trop bas inonde de bruit.

//gris clair 2//
**Decision log #2**
- Seuil initial: 0.90
- Itérations: 5
- Arrêt: si plus de candidats au‑dessus du seuil.

#### Établissez des prédictions fiables (Principe et algorithme)

Le pseudo-labeling consiste à prédire sur les données non étiquetées, sélectionner les prédictions au-dessus d’un seuil de confiance, les ajouter à l’entraînement comme labels temporaires, puis réentraîner le modèle. C’est la version « coach confiant » de l’apprentissage. 💬💪

```python
# Imports et préparation (PyTorch, MedMNIST)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import medmnist
from medmnist import INFO
from torchvision import transforms

data_flag = 'dermamnist'
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.5],[.5])])
train_dataset = DataClass(split='train', transform=transform, download=True)
test_dataset = DataClass(split='test', transform=transform, download=True)

# Sélectionner 50 images par classe (350 labels)
n_classes = len(info['label'])
labels_array = np.array(train_dataset.labels).flatten()
labeled_indices = []
for c in range(n_classes):
    idx = np.where(labels_array == c)[0]
    labeled_indices.extend(np.random.choice(idx, min(50, len(idx)), replace=False))
unlabeled_indices = list(set(range(len(train_dataset))) - set(labeled_indices))

labeled_loader = DataLoader(Subset(train_dataset, labeled_indices), batch_size=16, shuffle=True)
unlabeled_loader = DataLoader(Subset(train_dataset, unlabeled_indices), batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Petit CNN et boucle d'entraînement
class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(7*7*32, num_classes)
    def forward(self, x):
        x = self.layer1(x); x = self.layer2(x); x = x.view(x.size(0), -1); return self.fc(x)

def get_pseudo_labels(model, unlabeled_loader, threshold=0.9):
    model.eval(); indices, labels = [], []
    with torch.no_grad():
        for i, (images, _) in enumerate(unlabeled_loader):
            probs = torch.softmax(model(images), dim=1)
            max_p, preds = probs.max(dim=1)
            mask = max_p > threshold
            start = i * unlabeled_loader.batch_size
            batch_idx = torch.arange(start, start + images.size(0))
            original = [unlabeled_loader.dataset.indices[j] for j in batch_idx[mask]]
            indices.extend(original); labels.extend(preds[mask].tolist())
    return indices, labels
```

#### Découvrez un cas d’usage

Appliquez le pseudo-labeling sur `DermaMNIST` avec 350 images étiquetées. À chaque itération, ajoutez les plus confiantes des non étiquetées et réentraînez.

```python
device = torch.device('cpu')
model = SimpleCNN(in_channels=info['n_channels'], num_classes=n_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

current = list(labeled_indices)
for it in range(5):
    train_loader = DataLoader(Subset(train_dataset, current), batch_size=32, shuffle=True)
    # Entraîner quelques époques sur les labels courants...
    for epoch in range(5):
        model.train()
        for x, y in train_loader:
            y = y.squeeze().long()
            opt.zero_grad(); loss = criterion(model(x), y); loss.backward(); opt.step()
    # Générer des pseudo-labels
    pseudo_idx, pseudo_y = get_pseudo_labels(model, unlabeled_loader, threshold=0.9)
    for idx, yhat in zip(pseudo_idx, pseudo_y):
        if idx not in current:
            train_dataset.labels[idx] = yhat
            current.append(idx)
```

//rouge clair 3//
**Erreur classique**: baisser trop le seuil introduit du bruit et dégrade les performances. 🧯

#### Comprenez les avantages et les pièges du pseudo-labeling

- Avantages: simple, exploite immédiatement les données non étiquetées.
- Pièges: propagation d’erreurs, biais de confirmation, sensibilité au seuil et à la calibration.

#### À vous de jouer 🎯

Définissez les critères (métriques et seuils) pour décider si un pseudo-label est suffisamment fiable (ex. seuil de probabilité, marge top-1/top-2, calibration préalable) et justifiez vos choix.

#### En résumé 

* Le pseudo-labeling est un point d’entrée efficace et simple au SSL.
* La sélection par confiance et la calibration sont cruciales.
* Réentraînez par itérations en surveillant les métriques de test.

*Et si vos images se donnaient des tuyaux entre voisines ? Place aux graphes 🕸️*

### P1C4 : Modélisez les relations entre différentes entités 🕸️

//bleu clair 3//
**Note technique — Graphes et propagation**
- Embeddings via CNN (features): compacter l’information.
- Graphe k‑NN: relier les images proches.
- LabelSpreading: diffuser les labels connus vers les voisins.

//bleu clair 3//
**“Diagramme” (texte)**
- Nœuds = images, Arêtes = similarités
- Sources = labels connus → diffusion → labels pour les autres nœuds

//orange clair 3//
**Alerte**: embeddings faibles → propagation médiocre. Soignez l’extraction.

//gris clair 2//
**Decision log #3**
- Paramètres initiaux: k=10, kernel=knn
- Évaluation: AUC (OvR) + F1 macro sur non étiquetées

#### Découvrez le principe d’un graphe

On construit un graphe de similarité entre images (nœuds) et on propage l’information des quelques nœuds labellisés vers les autres. La sagesse des voisins au service de la prédiction ! 🧑‍🤝‍🧑

#### Propagez des labels aux nœuds non labellisés

```python
from sklearn.semi_supervised import LabelSpreading
from torch.utils.data import DataLoader

# Extraire des embeddings avec le CNN appris
def get_embeddings(model, dataset, device):
    model.eval(); embs = []
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    with torch.no_grad():
        for x, _ in loader:
            feats = model.layer2(model.layer1(x))
            feats = feats.view(feats.size(0), -1)
            embs.append(feats.numpy())
    import numpy as np; return np.vstack(embs)

embeddings = get_embeddings(model, train_dataset, device)
labels_for_spread = np.full(len(train_dataset), -1)
labels_for_spread[labeled_indices] = labels_array[labeled_indices]

spreader = LabelSpreading(kernel='knn', n_neighbors=10)
spreader.fit(embeddings, labels_for_spread)
pred_labels = spreader.transduction_
```

#### Découvrez un cas d’usage

Sur `DermaMNIST`, la propagation donne un étiquetage global cohérent. Comparez accuracy/F1 sur les exemples initialement non étiquetés pour juger de l’apport par rapport au pseudo-labeling.

#### À vous de jouer

Implémentez `LabelSpreading` sur vos données, puis évaluez la qualité de l’étiquetage obtenu. Quelles valeurs de `n_neighbors` et de noyau (rbf vs knn) fonctionnent le mieux ?

#### En résumé 

* Les graphes exploitent la structure globale des données. 🌐
* De bons embeddings améliorent fortement les résultats. 💪
* Validez sur des labels tenus à l’écart pour éviter la fuite d’information. 🔒

*Envie d'apprendre en jouant au chat et à la souris ? Direction les GANs 🎭*

### P1C5 : Classifiez des données non-étiquetées avec un générateur et un discriminateur 🎭

//bleu clair 3//
**Brief produit — SGAN**
- But: enrichir la représentation; D devient classificateur K+1.
- Données: étiquetées (supervisé), non étiquetées (réel), générées (faux).

//bleu clair 3//
**Note technique**
- D: K classes réelles + 1 “faux”.
- G: générer des “vraies‑semblances”.
- Risques: instabilité; monitorer pertes.

//bleu clair 3//
**Checklist QA — Hugo (MLOps)**
- Versionner seeds, checkpoints, courbes de pertes.
- Conserver un plan B (arrêt anticipé si instable).

//orange clair 3//
**Alerte**: n’investissez pas tout le temps ici si l’entraînement diverge. Priorité à une démo stable.

#### Découvrez le fonctionnement du générateur et du discriminateur (rappel GAN)

Un GAN oppose un Générateur (G) qui synthétise des images à un Discriminateur (D) qui distingue vrai/faux. En semi-supervisé (SGAN), D prédit à la fois la classe (0..K−1) pour les vraies images et une classe supplémentaire K pour les fausses.

#### Comprenez le rôle étendu du discriminateur (SGAN)

- Données étiquetées: perte supervisée sur la classe vraie.
- Données non étiquetées: encourager D à prédire une classe réelle (0..K−1) plutôt que la classe K.
- Images générées: D doit prédire la classe K (fausse).
- G cherche à produire des images que D classe comme réelles (0..K−1).

#### Utilisez des SGANs 🤝

Stratégie pratique: réutilisez votre pipeline de chargement (`labeled_loader`, `unlabeled_loader`) et entraînez D avec des têtes adaptées (`K+1` sorties). Évaluez D comme classificateur sur le test set et comparez aux méthodes précédentes.

#### À vous de jouer

Recherchez un exemple SGAN sur images médicales. Identifiez les composants clés (G, D, pertes) et expliquez comment les données étiquetées vs non étiquetées sont intégrées.

#### En résumé 

* Les SGANs tirent parti des données non étiquetées via la tâche auxiliaire vrai/faux.
* Le discriminateur devient un classificateur amélioré avec `K+1` classes.
* La stabilité d’entraînement et la qualité des images générées sont critiques.

*Stabilisons tout ça avec un peu de zen : la cohérence arrive 🔁*

### P1C6 : Maintenez des prédictions stables avec la régularisation par cohérence 🔁

//bleu clair 3//
**Playbook — Régularisation par cohérence**
- Vues faibles/fortes: mêmes labels attendus.
- Augmentations réalistes: rotations légères, contrastes.
- Seuil de confiance: générer des pseudo‑labels pour vues faibles (esprit FixMatch).

//bleu clair 3//
**Note clinique — Dr Malik**
- Éviter les transformations non plausibles en dermato.
- Viser une robustesse “démo‑proof”.

//bleu clair 3//
**Checklist démo**
- Latence acceptable
- Prédictions stables à perturbations mineures
- Visualisations lisibles pour le pitch

#### Comprenez le principe de la régularisation par cohérence

Objectif: des prédictions stables quand on applique de petites perturbations (augmentations) aux entrées. Exemples: Mean Teacher, UDA/FixMatch (consistance forte/faible). 🧘‍♀️

#### Découvrez un cas d’usage

En radiologie, la cohérence aux légères rotations/contrastes améliore la robustesse. On pénalise la divergence entre prédictions sur une image et sa version augmentée.

#### Appliquez des méthodes de perturbation des données

- Augmentations géométriques: rotation faible, flip, léger zoom.
- Augmentations photométriques: luminosité/contraste, jitter de couleur.
- Seuil de confiance pour générer des pseudo-labels sur les vues faibles (FixMatch).

#### À vous de jouer

Imaginez un pipeline de régularisation par cohérence adapté à vos images (types d’augmentations, intensités, seuils). Justifiez vos choix.

#### En résumé 

* La cohérence réduit le sur-apprentissage et améliore la calibration. 🎛️
* Combinez cohérence + pseudo-labeling pour de meilleurs résultats. 🧩
* Choisissez des augmentations réalistes par rapport au domaine. 🖼️

*Prêt pour le niveau expert ? On passe à la vitesse supérieure 🚀*

### P1C7 : Allez plus loin 🚀

> Email — Emma → stakeholders
>
> Objet: Démo POC — Synthèse et prochaines étapes
>
> Bonjour,
>
> Le POC d’aide au tri dermatologique est prêt pour la démo: résultats consolidés, documentation de nos décisions (thresholds, propagation, cohérence), et risques identifiés. Nous proposons une roadmap: améliorer les embeddings, intégrer FixMatch/FlexMatch, et cadrer une validation clinique.
>
//bleu clair 3//
**Documentation projet — Sommaire**
- Briefs produits
- Notes techniques (SSL, graphes, cohérence, SGAN)
- Decision logs (#1–#3)
- Checklists (QA, démo)
- Alerte & vigilance clinique

[TLK]

#### Techniques avancées

- FixMatch/FlexMatch: pseudo-labels à haute confiance + consistance forte/faible. 🧱
- Self-training avec calibration et stratégies d’échantillonnage par incertitude. 🎯
- Meilleurs extracteurs: ResNet/ViT pré-entraînés pour des embeddings plus discriminants. 🧠

#### Défis restants

- Robustesse aux outliers et au décalage de domaine. 🛡️
- Équité et biais en santé (distribution des classes, démographie). ⚖️
- Évaluation fiable quand les labels sont rares (protocoles d’annotation). 📏

#### Perspectives

- Intégration de l’auto-supervisé (SimCLR/BYOL) au SSL. 🔗
- Few-shot et méta-apprentissage pour données rares. 🧭
- Pipelines hybrides: graphes + consistance + pseudo-labels. 🧬

#### En résumé 