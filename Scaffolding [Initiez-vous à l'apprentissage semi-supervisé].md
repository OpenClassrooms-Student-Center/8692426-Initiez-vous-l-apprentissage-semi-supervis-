# \[Initiez-vous à l'apprentissage semi-supervisé\]

# **Calendrier de production**

**Rédaction du plan de cours :** du \[date\] au \[date\] pour validation le \[date\]

| Rédaction (chapitres & quiz) | Enregistrement des screencasts |
| :---- | :---- |
| Partie 1 : Livraison le \[date\] pour validation le \[date\] | \[date\] à \[date\] ou Tournage au studio \[16 septembre\] à \[10:00\].  ⇒ 17 bd Morland 75004 Paris |

**Congés SME & LCD :** 

* Laurène   
  * Du 25 août au 5 septembre  
* Aurélien  
  * Du 13 août au 10 septembre

**Liens utiles**

* Texte du cours *(Ajouter le lien vers le document)*  
* Conseil rédaction pour concevoir [le plan détaillé du cours](https://www.notion.so/openclassrooms/Le-plan-du-cours-317fde5f32b548308136c1bc9c688baa)  
* [Lien vers les autres ressources d'aide à la conception d'un cours](https://openclassrooms.notion.site/SME-resources-ca37d085cee1402eb4bec8f3411fbcd4) 

# **Scaffolding**

# **Partie 1 \- Mettez en place un modèle semi-superviséDécouvrez l’approche semi-supervisée**

|  *À remplir par l’auteur expert :* Résumer le contenu pour chaque section de chapitre.  |  |  |  | *Goals*  Remember Understand Apply Analyze Create Evaluate | *Components* |
| :---- | :---- | ----- | ----- | :---: | ----- |
|  **P1C1 : Tirez un maximum de ce cours** |  |  |  |  |   |
| Découvrez l’objectif du cours | Maîtriser les fondements et applications de l'apprentissage semi-supervisé dans le domaine de l'imagerie médicale. |  |  | Understand |  |
| Découvrez le projet fil rouge du cours | Développer un modèle de segmentation semi-supervisé pour l'analyse d'images radiologiques. |  |  | Remember |  |
| Rencontrez votre professeur |  |  |  | Remember |  **MEET YOUR TEACHER**  |
| (optionnel) Téléchargez la fiche résumé du cours |  |  |  | Remember |  **COURSE SUMMARY**   |
|  **P1C2 : Découvrez le principe de l’apprentissage semi supervisé** |  |  |  |  |  |
| Rappelez-vous des principes du supervisé vs non supervisé | Supervisé vs Non-Supervisé : Rappels et Limites |  |  | Remember |  **TALKING HEAD  STATIC GRAPHIC **  |
| Comprenez l’intérêt de l’apprentissage semi-supervisé | Pourquoi le Semi-Supervisé (SSL) ? Le défi des données labellisées |  |  | Understand |  |
| Comprenez les principes fondamentaux du semi-supervisé | Principes Fondamentaux de l'SSL |  |  | Understand |  |
| À vous de jouer \! | Identifiez trois situations concrètes en imagerie médicale où l'obtention de données labellisées est un frein majeur et où l'apprentissage semi-supervisé pourrait être une solution. Justifiez vos choix. |  |  | Apply |  **A VOUS DE JOUER** |
|  **P1C3 : Catégorisez l’inconnu avec la pseudo labellisation le pseudo labeling** |  |  |  |  |  |
| Établissez des prédictions fiables | Principe et Algorithme de Base : Prédire et se fier à ses propres prédictions fiables. |  |  | Understand |  **STATIC GRAPHIC** |
| Découvrez un cas d’usage | Application au Diagnostic Précoce : Détection de nodules pulmonaires ou microcalcifications mammaires à partir d'un petit jeu de données labellisées. |  |  | Understand | **SCREENCAST** |
| Comprenez les avantages et les pièges du pseudo-labeling | Avantages et Pièges du Pseudo-Labeling : propagation des erreurs, seuil de confiance. |  |  | Analyze |  |
| À vous de jouer \! | Vous développez un système de pseudo-labeling pour la détection de fractures par imagerie. Quels critères (métriques, seuils de confiance, etc.) mettriez-vous en place pour décider si un pseudo-label est suffisamment "fiable" pour être intégré à l'ensemble d'entraînement ? |  |  | Create |  **A VOUS DE JOUER** |
|  **P1C4 : Modélisez les relations entre différentes entités**  |  |  |  |  |  |
| Découvrez le principe d’un graphe | Introduction aux Graphes : Modéliser les relations entre différentes entités. |  |  | Remember |  **STATIC GRAPHIC** |
| Propagez des labels aux nœuds non labellisés | Propagation de Labels sur Graphes : Étendre les informations des nœuds labellisés aux nœuds non labellisés. |  |  | Understand |  **STATIC GRAPHIC** |
| Découvrez un cas d’usage | Cas d'usage : Classification automatique d’images médicales. |  |  | Analyze | **SCREENCAST** |
| À vous de jouer \! | À l'aide d'une bibliothèque comme scikit-learn, implémentez un algorithme de propagation d'étiquettes sur un petit ensemble d'images médicales (synthétiques ou réelles) pour segmenter une zone d'intérêt, en ne fournissant que quelques pixels étiquetés. Évaluez la qualité de la segmentation obtenue. |  |  | Create |  **A VOUS DE JOUER** |
|  **P1C5 : Classifiez des données non-étiquetées avec un générateur et un discriminateur** |  |  |  |  |  |
| Découvrez le fonctionnement du générateur et du discriminateur | Rappel sur les GANs : Fonctionnement du générateur et du discriminateur. |  |  | Remember |  **STATIC GRAPHIC** |
| Comprenez le rôle étendu du discriminateur | GANs et Apprentissage Semi-Supervisé (SGANs) : Le rôle étendu du discriminateur pour la classification de données non-étiquetées. |  |  | Understand |  **STATIC GRAPHIC** |
| Utilisez des SGANs | Utilisation des SGANs pour l'augmentation de données synthétiques et la classification en imagerie médicale. |  |  | Understand | **SCREENCAST** |
| Comprenez les avantages et les limites de ces modèles | Avantages et limites : La stabilité de l'entraînement et la qualité des données générées. |  |  | Analyze |  |
| À vous de jouer \! | Recherchez un exemple de code d'un SGAN appliqué à l'imagerie. Identifiez les composants clés (générateur, discriminateur, classification) et discutez de la façon dont le modèle gère les données étiquetées et non-étiquetées. |  |  | Apply |  **A VOUS DE JOUER** |
|  **P1C6 : Maintenez des prédictions stables avec la régularisation par cohérence** |  |  |  |  |  |
| Comprenez le principe de la régulation par cohérence | Comprendre la Régularisation par Cohérence (Mean Teacher, UDA) : Maintenir des prédictions stables malgré de petites perturbations des données. |  |  | Understand |  **STATIC GRAPHIC** |
| Découvrez un cas d’usage | Application à la segmentation de radios : utilisation des augmentations de données intelligentes. |  |  | Analyze | **SCREENCAST?** |
| Appliquer des méthodes de perturbation des données | Méthodes de Perturbation des Données : Augmentations géométriques et photométriques pertinentes pour l'imagerie médicale. |  |  | Understand | **SCREENCAST** |
| À vous de jouer \! | Imaginez un pipeline de régularisation par cohérence pour la segmentation automatique du des radios du chapitre précédent. Quelles types de perturbations (augmentations) appliqueriez-vous à vos images non labellisées pour que le modèle apprenne une représentation robuste ? |  |  | Analyze |  **A VOUS DE JOUER** |
|  **P1C7 : Allez plus loin** |  |  |  |  |  |
| Découvrez des techniques avancées | Vue d'ensemble de techniques avancées (ex: FixMatch, FlexMatch) : Combiner pseudo-labeling et régularisation par cohérence. |  |  | Understand |  **TALKING HEAD**  |
| Comprenez l’enjeu des défis restants | Les Défis Restants : Robustesse aux données aberrantes, gestion des biais, évaluation fiable des performances. |  |  | Understand |  |
| Projetez-vous sur les évolutions futures du SSL | Perspectives et Évolutions Futures en Imagerie Médicale : Apprentissage auto-supervisé intégré, few-shot learning, “méta-apprentissage” pour les données rares. |  |  | Remember |  |
| Quiz | **8 questions filées sur un cas d’usage \- sur un jeu de données** |  |  | Evaluate | **SCENARIO-BASED QUIZ**  |

# 