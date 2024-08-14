# Function to train the models
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback, BaseCallback
from feature_extractors import GTrXL_FeatureExtractor, CNN_1D_FeatureExtractor, CNN_LSTM_FeatureExtractor

def create_callbacks(eval_env, log_dir, reward_threshold=None, checkpoint_freq=10000):
    """
    Crée des callbacks pour le suivi de l'entraînement avec évaluation périodique.
    Args:
    eval_env (gym.Env): Environnement d'évaluation.
    log_dir (str): Répertoire pour enregistrer les logs.
    reward_threshold (float): Seuil de récompense pour arrêter l'entraînement si atteint.
    checkpoint_freq (int): Fréquence pour sauvegarder des checkpoints.
    Returns:
    list: Liste des callbacks.
    """
    os.makedirs(log_dir, exist_ok=True)

    class TensorboardCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(TensorboardCallback, self).__init__(verbose)

        def _on_step(self) -> bool:
            # Ajoutez ici toutes les métriques que vous souhaitez enregistrer
            value_loss = self.model.logger.name_to_value.get("train/value_loss", None)
            policy_loss = self.model.logger.name_to_value.get("train/policy_loss", None)
            entropy = self.model.logger.name_to_value.get("train/entropy_loss", None)

            # Enregistrez ces valeurs dans TensorBoard
            if value_loss is not None:
                self.logger.record("losses/value_loss", value_loss)
            if policy_loss is not None:
                self.logger.record("losses/policy_loss", policy_loss)
            if entropy is not None:
                self.logger.record("losses/entropy", entropy)

            return True

    # Callback pour sauvegarder des modèles à des intervalles réguliers
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_freq, save_path=log_dir,
                                             name_prefix='ppo_checkpoint')

    # Callback pour évaluer le modèle périodiquement et sauvegarder le meilleur
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=1000,
                                 deterministic=True, render=False)

    # Callback pour arrêter l'entraînement si la récompense atteint un certain seuil
    if reward_threshold is not None:
        stop_training_callback = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
        return [checkpoint_callback, eval_callback, TensorboardCallback(), stop_training_callback]
    else:
        return [checkpoint_callback, eval_callback, TensorboardCallback()]

# Function to train the PPO model
def train_ppo(env, eval_env, log_dir, total_timesteps=1000, n_steps=2048, batch_size=64, n_epochs=10, reward_threshold=None):
    """
    Entraîne un modèle PPO sur l'environnement de trading spécifié avec des callbacks.
    Args:
    env (gym.Env): L'environnement d'entraînement.
    eval_env (gym.Env): L'environnement de validation pour sauvegarder le meilleur modèle.
    log_dir (str): Répertoire pour enregistrer les logs et les modèles.
    total_timesteps (int): Nombre total de pas de temps pour l'entraînement.
    n_steps (int): Nombre de pas dans l'optimiseur.
    batch_size (int): Taille du lot pour l'entraînement.
    n_epochs (int): Nombre d'époques pour l'entraînement.
    reward_threshold (float): Seuil de récompense pour arrêter l'entraînement si atteint.
    Returns:
    stable_baselines3.PPO: Le modèle PPO entraîné.
    """
    # Paramètres spécifiques du modèle
    policy_kwargs = dict(
        features_extractor_class=GTrXL_FeatureExtractor  # Choisir votre extracteur de features
    )

    # Crée les callbacks
    callbacks = create_callbacks(eval_env, log_dir, reward_threshold=reward_threshold)

    # Entraîne le modèle
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        verbose=1,
        tensorboard_log=log_dir,  # Pour TensorBoard
    )

    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    return model
