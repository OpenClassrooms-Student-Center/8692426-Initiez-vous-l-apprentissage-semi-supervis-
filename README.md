# Initiez-vous à l'apprentissage semi-supervisé (SSL) 🚀

Apprenez les bases et les techniques avancées de l'apprentissage semi-supervisé à travers une série de notebooks guidés. Ce dépôt est pensé pour les étudiant·e·s curieux·ses qui veulent comprendre, expérimenter et s'amuser avec le SSL.

---

## Pourquoi le SSL ? 🤔

En apprentissage supervisé, on a besoin de beaucoup de données étiquetées… ce qui coûte cher. En non supervisé, on n'utilise pas les labels. Le semi-supervisé (SSL) combine le meilleur des deux mondes: il exploite un petit jeu de données labellisé et un grand jeu non labellisé pour améliorer les performances.

Idées clés que vous verrez ici:
- **Propagation de labels**: diffuser l'information des étiquettes via les voisins dans un graphe.
- **Régularisation par cohérence**: encourager le modèle à être stable sous perturbations (bruit, augmentations).
- **Pseudo-labeling**: utiliser les prédictions du modèle comme pseudo-labels sur les données non-labellisées.
- **Méthodes SOTA**: MixMatch, FixMatch, FlexMatch.
- **GANs semi-supervisés**: tirer parti des générateurs pour mieux classifier.

---

## Contenu du dépôt 📚

Les notebooks sont en français et indépendants. Parcourez-les dans l'ordre suggéré ou piochez selon vos besoins.

- `ssl_notebook.ipynb` — Panorama du SSL, principes, pipeline type, premières expériences.
- `La Propagation de Labels, ou l'art de juger une image par ses voisins.ipynb` — Label Propagation/Spreading, similarités, graphes, intuition et démos.
- `La Régularisation par Cohérence, ou l'art d'être constant avec soi-même.ipynb` — Consistency regularization, augmentations, objectifs de stabilité.
- `Le Pseudo-Labeling, ou l'art de faire confiance à son modèle.ipynb` — Pseudo-labels, seuils de confiance, itérations.
- `Advanced SSL Techniques - FixMatch, FlexMatch, and MixMatch.ipynb` — Implémentations et concepts des méthodes récentes.
- `Les Semi-Supervised GANs, ou l'art de générer pour mieux classer.ipynb` — Cadre GAN pour le SSL et exemples guidés.
- `dermamnist_ssl_model.pth` — Exemple de poids entraînés (DermMNIST) pour illustrer l'inférence/évaluation.

Note: Certains notebooks peuvent télécharger automatiquement des petits jeux de données ou expliquer comment les obtenir.

---

## Prérequis 🧰

- Python 3.9+ recommandé
- Jupyter (Notebook ou Lab)
- Bibliothèques usuelles: NumPy, pandas, scikit-learn, matplotlib/seaborn
- Deep learning: PyTorch ou TensorFlow selon les notebooks (la majorité est orientée PyTorch)

S'il n'y a pas de `requirements.txt`, installez au besoin depuis les erreurs import (les notebooks rappellent généralement les dépendances).

---

## Installation rapide ⚙️

Option 1 — Environnement virtuel minimal:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install jupyter numpy pandas scikit-learn matplotlib seaborn torch torchvision torchaudio
```

Option 2 — Conda:

```bash
conda create -n ssl python=3.10 -y
conda activate ssl
pip install jupyter numpy pandas scikit-learn matplotlib seaborn torch torchvision torchaudio
```

---

## Démarrage 🏁

1. Lancez Jupyter:
   ```bash
   jupyter lab
   # ou
   jupyter notebook
   ```
2. Ouvrez les notebooks dans le dossier: `.../8692426-Initiez-vous-l-apprentissage-semi-supervis-/`.
3. Exécutez les cellules dans l'ordre. Lisez les explications entre les blocs de code.

Astuce: si un import échoue, installez le paquet manquant avec `pip install <paquet>` puis relancez le noyau.

---

## Parcours d'apprentissage conseillé 🗺️

1. `ssl_notebook.ipynb` — Comprendre les motivations et le pipeline général.
2. Graph-based SSL — Propagation de labels.
3. Consistency regularization — Perturbations et objectifs de stabilité.
4. Pseudo-labeling — Auto-enseignement sur données non-labelisées.
5. Méthodes avancées — MixMatch, FixMatch, FlexMatch.
6. Bonus — GANs semi-supervisés.

Vous pouvez ensuite revenir sur vos jeux de données et adapter les techniques vues pour vos projets.

---

## Conseils pour de bons résultats 💡

- **Qualité des augmentations**: en SSL moderne (FixMatch/FlexMatch), les choix d'augmentations fortes sont déterminants.
- **Seuils de confiance**: ajustez-les pour contrôler le bruit des pseudo-labels.
- **Balance labellisé/non-labellisé**: surveillez les ratios dans les batchs.
- **Validation**: gardez un petit set validé pour suivre l'apprentissage sans fuite d'information.
- **Reproductibilité**: fixez les seeds quand vous comparez des méthodes.

---

## Dépannage 🛠️

- Problèmes CUDA/GPU: commencez en CPU en retirant `.to(device)`/`cuda()` ou en forçant `device='cpu'`.
- ImportError: installez le paquet manquant. Vérifiez les versions de PyTorch compatibles avec votre CUDA.
- Mémoire insuffisante: réduisez la taille des batchs et/ou les dimensions des images.

---

## Ressources pour aller plus loin 📖

- MixMatch: Beyond Empirical Risk Minimization (Berthelot et al., 2019)
- FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence (Sohn et al., 2020)
- FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling (Zhang et al., 2021)
- Semi-Supervised Learning (Chapelle, Scholkopf, Zien) — le classique

Ces références complètent les intuitions développées dans les notebooks.

---

## Remerciements 🙌

Ce dépôt a pour but de vous guider pas à pas. N'hésitez pas à expérimenter, casser, recommencer… c'est comme ça qu'on apprend ! Bon apprentissage semi-supervisé ✨