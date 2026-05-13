"""
Utilitaire de cache pour sauvegarder/charger les résultats intermédiaires
dans data/files/{img_name}_{étape}.npy
"""
import numpy as np
import os

# Chemin vers le dossier data/files, relatif à ce fichier
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, '../../data/files'))


def _get_path(img_name: str, step: str) -> str:
    """Construit le chemin vers le fichier cache pour une image et une étape."""
    return os.path.join(_DATA_DIR, f'{img_name}_{step}.npy')


def load_cache(img_name: str, step: str):
    """
    Charge un résultat depuis le cache s'il existe.
    Retourne le tableau numpy ou None si le fichier n'existe pas.
    """
    path = _get_path(img_name, step)
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None


def save_cache(img_name: str, step: str, data: np.ndarray):
    """Sauvegarde un résultat dans le cache."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    path = _get_path(img_name, step)
    np.save(path, data)


def get_img_name(img_path: str) -> str:
    """Extrait le nom de l'image (sans extension) à partir du chemin."""
    return os.path.splitext(os.path.basename(img_path))[0]
