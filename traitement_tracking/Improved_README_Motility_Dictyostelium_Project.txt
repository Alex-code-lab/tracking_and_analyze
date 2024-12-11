
# Projet d'Analyse de Motilité Dictyostelium

Ce fichier explique chaque script utilisé dans le projet, ses objectifs, les données d'entrée nécessaires, ainsi que des instructions détaillées pour son utilisation. Il inclut également les conventions d'organisation des dossiers et des fichiers indispensables au fonctionnement.

---

## Organisation des Dossiers et Fichiers

### Structure Générale
```
/Chemin/Principal/
    ├── Nom_Expérience_1/
    │      ├── Stack_Original/ (images d'origine au format TIFF)
    │      └── 8bits/ (images converties en 8 bits)
    ├── Nom_Expérience_2/
    │      ├── Stack_Original/
    │      └── 8bits/
    └── ...
```

- **Nom_Expérience_1/** : Un dossier unique par expérience, nommé avec :
  - Date : `YYYY_MM_DD`
  - Code d'expérience : `ASMOTxxx`
  - Description : informations supplémentaires (cellules, objectif, etc.)
  Exemple : `2024_03_26_ASMOT143_AX3`.

- **Stack_Original/** : Contient les images originales au format `.tif`, organisées en sous-dossiers.

- **8bits/** : Généré automatiquement après conversion, contient les images en 8 bits.

---

### Conventions de Nom des Fichiers
- Les fichiers TIFF doivent être séquentiellement nommés pour garantir un ordre correct.
  Exemple :
  ```
  img_000001.tif
  img_000002.tif
  ```

---

## Scripts et Instructions

### **0_Redimension_frames.ipynb**
**Objectif :** Préparer les images pour s'assurer qu'elles ont des dimensions uniformes.  
**Données nécessaires :**
- Images `.tif` ou `.png`.

**Utilisation :**
1. Spécifiez le dossier contenant les images avec `PATHWAY_PICTURES`.
2. Lancez le script pour produire des images redimensionnées.

---

### **1_Convertisseur_8bits_jupyter.ipynb**
**Objectif :** Convertir les images originales en 8 bits.  
**Données nécessaires :**
- Dossiers contenant des images `.tif`.

**Utilisation :**
1. Configurez `general_path` vers le répertoire des expériences.
2. Le script génère un dossier `8bits/` contenant les images converties.

---

### **2-Making_mosaic.ipynb**
**Objectif :** Combiner plusieurs images pour créer des mosaïques.  
**Données nécessaires :**
- Dossiers `8bits/` générés à l'étape précédente.

**Utilisation :**
1. Configurez `position_folders` vers les dossiers d'images.
2. Les mosaïques sont générées dans un sous-dossier `mosaic/`.

---

### **3_Substrack_background.ipynb**
**Objectif :** Soustraire l’arrière-plan des mosaïques.  
**Données nécessaires :**
- Mosaïques générées précédemment.

**Utilisation :**
1. Définissez `path` vers le dossier contenant les mosaïques.
2. Lancez le script pour produire des images nettoyées.

---

### **4_Pre-test_analyse_images.ipynb**
**Objectif :** Configurer les paramètres d’analyse.  
**Données nécessaires :**
- Images nettoyées.

**Utilisation :**
1. Ajustez les paramètres dans `PARAMS`.
2. Testez les paramètres pour ajuster les seuils de détection.

---

### **5_Tracking.ipynb**
**Objectif :** Suivre les particules dans les images.  
**Données nécessaires :**
- Images nettoyées.

**Utilisation :**
1. Listez les expériences dans `EXPERIMENT_NAMES`.
2. Configurez les paramètres dans `PARAMS`.
3. Exécutez le script pour obtenir des trajectoires sous forme de fichiers HDF5.

---

### **6_trajectories_finales.ipynb**
**Objectif :** Nettoyer et préparer les trajectoires pour l’analyse.  
**Données nécessaires :**
- Fichiers HDF5 contenant les trajectoires.

**Utilisation :**
1. Définissez le chemin vers les résultats avec `PATHWAY_RESULTS`.
2. Ajustez les paramètres de nettoyage.

---

### **7-Analyze_mosaic_notebook_CytoOne_HL5.ipynb**
**Objectif :** Analyser les métriques extraites des trajectoires.  
**Données nécessaires :**
- Fichiers consolidés HDF5.

**Utilisation :**
1. Importez les données dans un DataFrame.
2. Réalisez des analyses et visualisations personnalisées.

---

### **8-Comparaison_between_conditions.ipynb**
**Objectif :** Comparer les métriques entre conditions expérimentales.  
**Données nécessaires :**
- Fichiers HDF5 spécifiques à chaque condition.

**Utilisation :**
1. Configurez le dossier contenant les fichiers avec `folder_path`.
2. Filtrez les données avec des mots-clés pour une analyse comparative.

---

### **9-creation_video_trajectoires.ipynb**
**Objectif :** Créer des vidéos pour illustrer les trajectoires.  
**Données nécessaires :**
- Trajectoires HDF5.
- Mosaïques correspondantes.

**Utilisation :**
1. Configurez les paramètres de trajectoires et frames.
2. Utilisez `creating_gif` pour générer des vidéos.

---

## Notes
- **Dépendances** : Assurez-vous d'installer toutes les bibliothèques (`pandas`, `numpy`, `matplotlib`, etc.).
- **Formats** : Vérifiez que les fichiers sont au bon format avant de lancer les scripts.

Pour toute question, contactez l'auteur.
