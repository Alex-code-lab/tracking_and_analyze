# Projet d'Analyse de Motilité Dictyostelium

Ce projet a pour objectif d’analyser la motilité des cellules Dictyostelium à partir d’images obtenues par microscopie. La chaîne de traitement se décline en plusieurs étapes allant de la préparation des images brutes à la génération de vidéos illustrant les trajectoires. Chaque étape est assurée par un notebook spécifique.

---

## Organisation des Dossiers

Pour que le traitement se déroule correctement, organisez vos dossiers de la manière suivante :

```bash
/Chemin/Principal/
├── Nom_Expérience_1/
│      ├── Stack_Original/   ← Images originales au format TIFF
│      ├── 8bits/            ← Images converties en 8 bits (générées par le script 1_Convertisseur_8bits_jupyter.ipynb)
│      └── mosaic/           ← Mosaïques créées à partir des images 8 bits (générées par le script 2-Making_mosaic.ipynb)
├── Nom_Expérience_2/
│      ├── Stack_Original/
│      ├── 8bits/
│      └── mosaic/
└── …
```

**Convention de nommage :**  
Chaque dossier d’expérience doit suivre la nomenclature suivante :  
`YYYY_MM_DD_ASMOTxxx_Description`  
*(exemple : `2024_03_26_ASMOT143_AX3`)*

---

## Liste des Scripts et Leur Utilisation

### 0_Redimension_frames.ipynb

**Objectif :**  
- Vérifier et redimensionner les images afin d’assurer des dimensions uniformes pour le traitement.

**Entrées :**  
- **Images d'origine :**  
  - Un dossier contenant les images (TIFF, PNG, etc.) à traiter.
- **Paramètres de redimensionnement :**  
  - Dimensions cibles (largeur, hauteur) et options éventuelles de recadrage ou de padding.

**Points à modifier :**  
- **Chemin d'accès :**  
  - Définissez la variable `PATHWAY_PICTURES` pour pointer vers le dossier contenant vos images.
- **Dimensions cibles :**  
  - Ajustez les paramètres de redimensionnement en fonction des exigences de votre analyse.

**Sorties attendues :**  
- Des images redimensionnées sauvegardées dans un dossier dédié ou en remplacement des originaux.
- Des logs ou visualisations confirmant la réussite du redimensionnement.

---

### 1_Convertisseur_8bits_jupyter.ipynb

**Objectif :**  
- Convertir les images originales en images 8 bits pour faciliter leur traitement ultérieur.

**Entrées :**  
- **Images d'origine :**  
  - Un dossier contenant les images originales (souvent en TIFF).
- **Paramètres :**  
  - Le chemin général des expériences (`general_path`) qui doit être défini.

**Points à modifier :**  
- **Chemin d'accès :**  
  - Configurez `general_path` pour pointer vers le dossier parent regroupant toutes les expériences.
- **Options de conversion :**  
  - Vérifiez que les options de conversion (par exemple, conversion en niveaux de gris) correspondent à vos besoins.

**Sorties attendues :**  
- La création automatique d’un sous-dossier `8bits/` dans chaque dossier d’expérience.
- Les images converties en 8 bits, prêtes pour la suite du traitement.

---

### 2-Making_mosaic.ipynb

**Objectif :**  
- Combiner plusieurs images 8 bits pour créer une mosaïque qui offre une vue globale de l’expérience.

**Entrées :**  
- **Images d'entrée :**  
  - Les images contenues dans les dossiers `8bits/` générés lors de l'étape précédente.
- **Paramètres :**  
  - Le chemin des dossiers de positions (`positions_dir`) où se trouvent les images.
  - Les options de concaténation pour organiser les images en mosaïque (par exemple, nombre de colonnes et rangées).

**Points à modifier :**  
- **Chemin d'accès :**  
  - Configurez `positions_dir` pour pointer vers le dossier contenant les images 8 bits.
- **Méthode de mosaïque :**  
  - Choisissez entre les différentes fonctions disponibles (comme `create_mosaic_images`, `create_mosaic_images_too_heavy` ou `create_mosaic_images_agregat`) selon les contraintes de performance et de mémoire.
- **Options de sauvegarde :**  
  - Définissez `output_dir` pour spécifier le dossier de sortie des mosaïques.

**Sorties attendues :**  
- Des mosaïques générées et sauvegardées dans un sous-dossier `mosaic/` pour chaque code de temps.
- Des messages ou logs confirmant la création réussie des mosaïques.

---

### 3_Substrack_background.ipynb

**Objectif :**  
- Soustraire l’arrière-plan des mosaïques afin d’améliorer la détection des particules (cellules).

**Entrées :**  
- **Images d'entrée :**  
  - Les mosaïques créées lors de l'étape précédente.
- **Paramètres :**  
  - Le chemin vers le dossier contenant les mosaïques.
  - Les options de traitement pour la soustraction de l’arrière-plan (méthode de filtrage, seuils, lissage, etc.).

**Points à modifier :**  
- **Chemin d'accès :**  
  - Vérifiez et ajustez le chemin vers les mosaïques à traiter.
- **Paramètres de traitement :**  
  - Ajustez les options (par exemple, le type de filtre ou le seuil) pour optimiser la soustraction de l’arrière-plan en fonction du bruit et de la qualité des images.

**Sorties attendues :**  
- Des images « nettoyées » (avec l’arrière-plan soustrait) prêtes à être utilisées pour la détection et le suivi des particules.
- Des visualisations ou logs démontrant l’efficacité de la soustraction de l’arrière-plan.

---

### 4_Pre-test_analyse_images.ipynb

**Objectif :**  
- Tester et ajuster les paramètres d’analyse d’images (floutage, seuils, etc.) pour optimiser la détection des particules.

**Entrées :**  
- **Images d'entrée :**  
  Un ensemble d’images prétraitées (par exemple, après soustraction de l’arrière-plan) qui représente un échantillon représentatif de l’expérience.
- **Paramètres :**  
  Le dictionnaire `PARAMS` contient les réglages suivants (entre autres) :
  - `GaussianBlur` : Tuple définissant la taille du filtre de flou (ex. `(5, 5)`).
  - `sigmaX` et `sigmaY` : Coefficients influençant l'effet du flou.
  - `threshold` : Valeur seuil pour la détection des particules.
  - `percentile` : Pour ajuster la distribution des valeurs d’intensité.
  - D’autres paramètres optionnels comme `smoothing_size`, `invert`, etc.

**Points à modifier :**  
- **Valeurs dans `PARAMS` :**  
  - Ajuster `GaussianBlur`, `sigmaX` et `sigmaY` pour obtenir le floutage désiré.
  - Modifier `threshold` pour filtrer le bruit sans perdre les particules réelles.
  - Tester différents réglages de `percentile` pour optimiser la détection.
- **Plage de frames :**  
  - Modifier `lenght_study` pour travailler sur un sous-ensemble de frames pendant la phase de test, afin d’améliorer les performances.

**Sorties attendues :**  
- Des visualisations (plots, images annotées) montrant l’impact des réglages sur la détection des particules.
- Des statistiques préliminaires (nombre de particules détectées, répartition des intensités, etc.) permettant de valider les paramètres optimaux.

---

### 5_Tracking.ipynb

**Objectif :**  
- Suivre les particules dans les images et extraire leurs trajectoires en utilisant la bibliothèque `trackpy`.

**Entrées :**  
- **Images d'entrée :**  
  Images prétraitées ou résultats de la détection issus de l’étape précédente.
- **Paramètres de suivi :**  
  Le dictionnaire `PARAMS` inclut notamment :
  - `diameter` : Diamètre estimé des particules.
  - `max_displacement` : Déplacement maximal autorisé entre deux frames.
  - `search_range` : Plage de recherche pour l’association des particules entre les frames.
  - D’autres paramètres comme `minmass`, `max_size`, `separation`, etc.
- **Condition expérimentale :**  
  La variable `CONDITION` (exemple : `'CytoOne_HL5_AMPC_10x'`) qui identifie l’expérience.

**Points à modifier :**  
- **Paramètres de suivi :**  
  - Adapter `diameter` selon la taille réelle des particules.
  - Ajuster `max_displacement` et `search_range` en fonction de la vitesse des particules et de la fréquence d’acquisition.
- **Chemins d’accès :**  
  - Vérifier et modifier si nécessaire les chemins vers les images (`data_dir`) et le dossier de sortie (`output_dir`).

**Sorties attendues :**  
- Des fichiers de trajectoires (au format HDF5 ou CSV) contenant, pour chaque particule, les positions (x, y) par frame.
- Des visualisations intermédiaires (par exemple, des plots montrant les trajectoires superposées aux images) pour valider le suivi.

---

### 6_trajectories_finales.ipynb

**Objectif :**  
- Nettoyer et préparer les trajectoires pour l’analyse ultérieure.
- Appliquer des corrections telles que la soustraction du drift global.

**Entrées :**  
- **Fichiers de trajectoires :**  
  Les résultats bruts issus du tracking.
- **Paramètres de correction :**  
  Paramètres (par exemple, la valeur de `smooth`) pour corriger le drift.

**Points à modifier :**  
- **Chemin des résultats :**  
  Spécifier correctement le chemin (`PATHWAY_RESULTS`) vers les trajectoires brutes.
- **Paramètres de correction :**  
  - Ajuster la valeur de `smooth` (ou tout autre paramètre pertinent) pour obtenir une estimation réaliste du drift.
  - Configurer les options de sauvegarde du plot (paramètres `save` et `pathway_saving`) si nécessaire.
- **Options de visualisation :**  
  Modifier la taille des figures, les limites des axes, ou le format d’enregistrement (`img_type`).

**Sorties attendues :**  
- Des fichiers contenant les trajectoires corrigées (avec le drift retiré).
- Des graphiques affichant les courbes de drift avant et après correction, permettant de vérifier l’efficacité du traitement.

---

### 7-Analyze_mosaic_notebook_CytoOne_HL5.ipynb *(et variantes)*

**Objectif :**  
- Analyser les métriques extraites des trajectoires (vitesse, direction, etc.) et générer des visualisations statistiques.

**Entrées :**  
- **Données de trajectoires :**  
  Fichiers consolidés issus des trajectoires corrigées.
- **Paramètres d’analyse :**  
  Critères et seuils pour l’analyse, tels que :
  - Seuils pour la vitesse minimale et maximale.
  - Critères pour exclure des trajectoires trop courtes ou bruitées.

**Points à modifier :**  
- **Import des données :**  
  Vérifier le chemin d’accès aux fichiers de trajectoires.
- **Filtres et seuils :**  
  Adapter les filtres appliqués (par exemple, en excluant les trajectoires avec moins de `min_frames` frames).
- **Options de visualisation :**  
  Ajuster le style des graphiques (taille, couleurs, labels) selon vos besoins.
- **Paramètres conditionnels :**  
  Modifier les réglages pour chaque condition expérimentale si vous analysez plusieurs jeux de données.

**Sorties attendues :**  
- Des graphiques statistiques (histogrammes, boxplots, courbes de tendance, etc.) illustrant la répartition des vitesses et des directions.
- Des tableaux synthétiques résumant les statistiques clés (vitesse moyenne, dispersion, etc.).

---

### 8-Comparaison_between_conditions.ipynb

**Objectif :**  
- Comparer les métriques de motilité entre différentes conditions expérimentales (par exemple, traitement vs. contrôle).

**Entrées :**  
- **Données de trajectoires ou statistiques synthétiques :**  
  Pour chaque condition expérimentale.
- **Chemins d’accès :**  
  Dossiers contenant les résultats de chaque condition.

**Points à modifier :**  
- **Chemins d’accès :**  
  Vérifier et ajuster les chemins pour chaque condition (par exemple, un dossier pour le traitement et un autre pour le contrôle).
- **Filtres et critères de sélection :**  
  Adapter les filtres pour comparer des ensembles de données homogènes.
- **Paramètres de visualisation :**  
  Modifier les options graphiques (couleurs, légendes, type de diagramme) pour mettre en évidence les différences.

**Sorties attendues :**  
- Des graphiques comparatifs (bar charts, boxplots, scatter plots) montrant les différences statistiques entre les conditions.
- Des tableaux ou rapports synthétiques résumant les métriques clés par condition.

---

### 9-creation_video_trajectoires.ipynb

**Objectif :**  
- Créer des vidéos ou des GIFs illustrant les trajectoires des particules superposées aux images de mosaïque, afin de visualiser leur évolution dans le temps.

**Entrées :**  
- **Fichiers de trajectoires :**  
  Résultats du tracking et trajectoires corrigées.
- **Images de mosaïques :**  
  Pour chaque frame, sur lesquelles les trajectoires seront superposées.
- **Paramètres vidéo :**  
  - Plage de frames à utiliser (ex. `first_frame` et `last_frame`).
  - Fréquence (FPS) et résolution de sortie.
  - Format de sortie (GIF, MP4, etc.).

**Points à modifier :**  
- **Plage de frames :**  
  Définir correctement `first_frame` et `last_frame` pour délimiter l’animation.
- **Paramètres vidéo :**  
  Ajuster la fréquence d’images (FPS) et la résolution en fonction des images et des exigences de l’analyse.
- **Chemins d’accès :**  
  Vérifier que les chemins vers les mosaïques et les trajectoires sont corrects.
- **Options de superposition :**  
  Configurer l'affichage des trajectoires (couleurs, épaisseur des lignes, labels éventuels).

**Sorties attendues :**  
- Un fichier vidéo ou un GIF animé montrant les trajectoires superposées aux images de mosaïque, illustrant l’évolution des particules au fil du temps.
- Des logs ou aperçus indiquant la progression et la confirmation de la sauvegarde dans le dossier de sortie.

---

## Instructions Générales

1. **Installation des dépendances :**  
   Assurez-vous d’installer toutes les bibliothèques requises (par exemple, `pandas`, `numpy`, `matplotlib`, `trackpy`, `scikit-image`, etc.) via `pip` ou `conda`.

2. **Organisation des dossiers :**  
   - Placez chaque expérience dans un dossier respectant le format `YYYY_MM_DD_ASMOTxxx_Description`.
   - Chaque dossier d’expérience doit contenir un sous-dossier `Stack_Original/` avec les images brutes.
   - Les scripts se chargeront de créer les dossiers `8bits/` et `mosaic/` nécessaires aux étapes suivantes.

3. **Ordre d’exécution des scripts :**  
   Pour garantir un traitement cohérent, suivez l’ordre suivant :
   1. **0_Redimension_frames.ipynb**
   2. **1_Convertisseur_8bits_jupyter.ipynb**
   3. **2-Making_mosaic.ipynb**
   4. **3_Substrack_background.ipynb**
   5. **4_Pre-test_analyse_images.ipynb**
   6. **5_Tracking.ipynb**
   7. **6_trajectories_finales.ipynb**
   8. **7-Analyze_mosaic_notebook_CytoOne_HL5.ipynb** *(et variantes si besoin)*
   9. **8-Comparaison_between_conditions.ipynb**
   10. **9-creation_video_trajectoires.ipynb**

4. **Personnalisation des paramètres :**  
   Chaque notebook comporte un dictionnaire de paramètres (`PARAMS`) que vous pouvez adapter à votre jeu de données. Il est conseillé de tester ces paramètres sur un petit échantillon d’images avant de lancer l’analyse complète.

5. **Sauvegarde et suivi des résultats :**  
   Les résultats (images converties, mosaïques, trajectoires, vidéos, etc.) sont sauvegardés automatiquement dans les dossiers correspondants à chaque expérience. Vérifiez régulièrement ces dossiers pour suivre l’avancement du traitement.

---

Ce README présente une vue d’ensemble détaillée pour l’utilisation des différents scripts du projet.  
Si vous avez des questions ou rencontrez des problèmes, consultez les commentaires dans chaque notebook ou contactez l’auteur pour plus d’informations.
