""" trading environment"""
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from data import preprocess_data

# Function to create the trading environment with monitoring
def reward_function(history):
    """ Simple reward for long/short """
    # Calcul de l'augmentation ou de la diminution de la valeur du portefeuille
    portfolio_growth = np.log(history['portfolio_valuation', -1] / history['portfolio_valuation', -2])

    # Pénalité pour changements fréquents de position
    change_penalty = -1 if history['position', -1] != history['position', -2] else 0

    # Calcul de la récompense totale
    total_reward = portfolio_growth + 0.001 * change_penalty
    return total_reward


def create_multi_env(dataset_dir, log_dir=None, state_window_size=100, max_episode_duration='max', short=False, n_envs=10):
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
            reward_function=reward_function,
            verbose=0
        )

    if log_dir is not None:
        def make_env_with_monitor():
            env = make_env()
            return Monitor(env, log_dir)
        envs = make_vec_env(make_env_with_monitor, n_envs=n_envs)
    else:
        envs = make_vec_env(make_env, n_envs=n_envs)
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
        reward_function=reward_function,
        verbose=0
    )

    if log_dir is not None:
        no_vec_env = Monitor(no_vec_env, log_dir)

    return no_vec_env
