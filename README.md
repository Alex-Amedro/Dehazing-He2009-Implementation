# Dehazing — Implémentation de He et al. (2009)

Implémentation en Python de l'algorithme de suppression de brume (*Single Image Haze Removal Using Dark Channel Prior*) proposé par He, Sun et Tang (CVPR 2009).

## Principe de l'algorithme

L'algorithme repose sur le **Dark Channel Prior** : dans les images naturelles sans brume, la plupart des patchs locaux contiennent au moins un pixel dont l'intensité est très faible sur au moins un canal couleur.

Le pipeline se décompose en 5 étapes :

1. **Dark Channel** — Calcul du minimum local sur les 3 canaux couleur
2. **Lumière atmosphérique (A)** — Estimation à partir des pixels les plus brillants du dark channel
3. **Transmission (t̃)** — Estimation brute de la carte de transmission
4. **Soft Matting** — Raffinement de la transmission via la matting Laplacian (Levin et al.)
5. **Reconstruction (J)** — Restitution de l'image sans brume via `J = (I - A) / max(t, t₀) + A`

Un module bonus permet de calculer une **carte de profondeur** à partir de la transmission : `d = -log(t)`.

## Utilisation

### Pipeline complet (reconstruction)

```bash
cd src/functions
python image_reconstruction.py <chemin_image> [options]
```

**Exemple :**

```bash
python image_reconstruction.py ../../data/raw_images/forest.jpg --patch-size 15 --omega 0.95
```

Affiche 3 panneaux : image originale | carte de transmission | image sans brume.

### Étapes individuelles

Chaque module peut être exécuté indépendamment :

```bash
# Dark Channel
python darkchannelcomputation.py ../../data/raw_images/forest.jpg --patch-size 20

# Lumière atmosphérique
python atmoslight.py ../../data/raw_images/forest.jpg --patch-size 20

# Estimation de transmission
python transmission_estimation.py ../../data/raw_images/forest.jpg --patch-size 20 --omega 0.95

# Soft Matting (raffinement)
python soft_matting.py ../../data/raw_images/forest.jpg --patch-size 20 --omega 0.95 --lambda-e 0.0001

# Carte de profondeur
python depth_map.py ../../data/raw_images/forest.jpg --patch-size 20 --omega 0.95
```

### Paramètres

| Paramètre       | Défaut   | Description                                                                 |
|------------------|----------|-----------------------------------------------------------------------------|
| `image`          | —        | Chemin vers l'image d'entrée (obligatoire)                                  |
| `--patch-size`   | `20`     | Taille du patch pour le filtre minimum du dark channel                      |
| `--omega`        | `0.95`   | Coefficient de dévoilement (0 = pas de dévoilement, 1 = dévoilement total) |
| `--t0`           | `0.1`    | Seuil minimal de transmission (évite la sur-amplification du bruit)        |
| `--lambda-e`     | `0.0001` | Régularisation du soft matting (plus grand = plus proche de t̃ brut)        |

## Système de cache

Les résultats intermédiaires sont automatiquement sauvegardés dans `data/files/` au format `.npy` pour éviter de recalculer les étapes coûteuses (en particulier le soft matting).

- Nommage : `{nom_image}_{étape}.npy` (ex : `forest_dark_channel.npy`)
- Le cache est chargé automatiquement si le fichier existe

** Important :** Si vous modifiez les paramètres (patch size, omega…) ou le code, pensez à supprimer le cache correspondant pour forcer un recalcul.



## Structure du projet

```
Dehazing-He2009-Implementation/
├── data/
│   ├── raw_images/          # Images d'entrée
│   └── files/               # Cache des résultats intermédiaires (.npy)
├── src/
│   └── functions/
│       ├── darkchannelcomputation.py    # Étape 1 : Dark Channel Prior
│       ├── atmoslight.py               # Étape 2 : Lumière atmosphérique
│       ├── transmission_estimation.py   # Étape 3 : Estimation de la transmission
│       ├── soft_matting.py              # Étape 4 : Raffinement (matting Laplacian)
│       ├── image_reconstruction.py      # Étape 5 : Reconstruction finale
│       ├── depth_map.py                 # Bonus : Carte de profondeur
│       └── cache_utils.py              # Utilitaire de cache
├── requirements.txt
└── README.md
```

## Référence

> K. He, J. Sun, and X. Tang, *"Single Image Haze Removal Using Dark Channel Prior"*, IEEE CVPR, 2009.
