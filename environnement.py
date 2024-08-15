import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from data import preprocess_data

# Function to create the trading environment with monitoring
def reward_function_both(history):
    # Calcul de l'augmentation ou de la diminution de la valeur du portefeuille
    portfolio_growth = np.log(history['portfolio_valuation', -1] / history['portfolio_valuation', -2])

    # Pénalité pour changements fréquents de position
    change_penalty = -1 if history['position', -1] != history['position', -2] else 0

    # Pénalité pour ne pas suivre la tendance du marché
    if history['position', -1] == 0 and history['data_close', -1] > history['data_close', -2]:
        missed_opportunity_penalty = -np.log(history['data_close', -1] / history['data_close', -2])
    elif history['position', -1] == 0 and history['data_close', -1] < history['data_close', -2]:
        missed_opportunity_penalty = np.log(history['data_close', -1] / history['data_close', -2])
    else:
        missed_opportunity_penalty = 0


    # Calcul de la récompense totale
    total_reward = portfolio_growth + 0.001 * change_penalty + 0.3 * missed_opportunity_penalty

    return total_reward

def reward_function_long(history):
    # Historique des transactions passées
    last_trade = None

    # Recherche du dernier ordre de trading
    for i in range(-2, -len(history['position'])-1, -1):
        if history['position', i] != history['position', i+1]:
            last_trade = i
            break

    # S'il n'y a pas d'ordre précédent, aucune récompense ne peut être calculée
    if last_trade is None:
        return 0

    # Vente (passage de la position de 1 à 0)
    if history['position', last_trade] == 1 and history['position', -1] == 0:
        prix_vente = history['data_close', -1]
        prix_achat = history['data_close', last_trade]
        quantite = history['portfolio_distribution_asset', last_trade]
        # Calcul du gain lors de la vente
        gain = (prix_vente*0.992 - prix_achat) * quantite
        # print(f"buy asset:{history['portfolio_distribution_asset', last_trade]:.2f}, buy/sell:{prix_vente*.992:.2f}/{prix_achat:.2f}={gain/quantite*100:.2}%, quantite:{quantite:.2f}, gain:{gain:.2f}")

        return gain

    # Achat (passage de la position de 0 à 1)
    elif history['position', last_trade] == 0 and history['position', -1] == 1:
        prix_achat = history['data_close', -1]
        prix_vente_prec = history['data_close', last_trade]
        quantite = history['portfolio_distribution_fiat', last_trade] / prix_achat

        # Calcul du gain lors de l'achat (comme un short)
        gain = (prix_vente_prec - prix_achat*1.002) * quantite
        # print(f"sell asset:{history['portfolio_distribution_fiat', last_trade]:.2f}, sell/buy:{prix_vente_prec:.2f}/{prix_achat*1.002:.2}={(prix_vente_prec - prix_achat*1.002)*100:.1}%, quantite:{quantite:.2f}, gain:{gain:.2f}")

        return gain

    # Pas de changement de position ou situation non couverte
    return 0

def create_multi_env(dataset_dir, log_dir=None, state_window_size=100, max_episode_duration='max', short=False):
    """
    Crée un environnement de trading pour chaque ensemble de données dans un répertoire.
    Args:
    dataset_dir (str): Chemin vers les fichiers de données dans un répertoire spécifique.
    log_dir (str): Répertoire pour enregistrer les logs (facultatif).
    state_window_size (int): Taille de la fenêtre des états pour l'environnement.
    max_episode_duration (str or int): Durée maximale de l'épisode.
    short (bool): Si vrai, permet les positions courtes.
    Returns:
    gym.Env: L'environnement de trading créé.
    """
    # def make_env():
    def make_env():
        return gym.make(
            "MultiDatasetTradingEnv",
            dataset_dir=dataset_dir,
            preprocess=preprocess_data,
            positions=[0, 1] if short is False else [-1, 0, 1],  # SHORT, OUT, LONG
            trading_fees=0.01 / 100,  # 0.01% fees
            windows=state_window_size,
            max_episode_duration=max_episode_duration,
            reward_function=reward_function_long if not short else reward_function_both,
        )

    if log_dir is not None:
        def make_env_with_monitor():
            env = make_env()
            return Monitor(env, log_dir)
        envs = make_vec_env(make_env_with_monitor, n_envs=10)
    else:
        envs = make_vec_env(make_env, n_envs=10)
    return envs

def create_env(dataset_path, log_dir=None, state_window_size=100, max_episode_duration='max', short=False):
    """
    Crée un environnement de trading de test pour un pkl.
    Args:
    dataset_path (str): Chemin vers les fichiers de données dans un répertoire spécifique.
    log_dir (str): Répertoire pour enregistrer les logs (facultatif).
    state_window_size (int): Taille de la fenêtre des états pour l'environnement.
    max_episode_duration (str or int): Durée maximale de l'épisode.
    short (bool): Si vrai, permet les positions courtes.
    Returns:
    gym.Env: L'environnement de trading créé.
    """
    data = pd.read_pickle(dataset_path)

    data = preprocess_data(data)

    no_vec_env = gym.make(
        "TradingEnv",
        df=data,
        positions=[0, 1] if short is False else [-1, 0, 1],  # SHORT, OUT, LONG
        trading_fees=0.01 / 100,  # 0.01% fees
        windows=state_window_size,
        max_episode_duration=max_episode_duration,
        reward_function=reward_function_long if not short else reward_function_both
    )

    if log_dir is not None:
        no_vec_env = Monitor(no_vec_env, log_dir)

    return no_vec_env
