""" Function to train the models """
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from feature_extractors import GTrXL_FeatureExtractor, CNN_1D_FeatureExtractor, CNN_LSTM_FeatureExtractor
class TensorboardCallback(BaseCallback):
    """ Call back """
    def __init__(self, train_env, eval_env, test_env, log_dir, eval_freq=10000, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env
        self.test_env = test_env
        self.log_dir = log_dir
        self.best_ratio = -float('inf')  # Pour sauvegarder le meilleur modèle
        self.eval_freq = eval_freq
        self.last_eval_step = 0

    def _on_step(self):
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

        # Ajoutez ici toutes les métriques que vous souhaitez enregistrer
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            print(f'num_timesteps:{self.num_timesteps}')
            self.last_eval_step = self.num_timesteps
            # Calcul des retours pour l'environnement d'entraînement
            train_portfolio_return, train_market_return, train_ratio = self._calculate_portfolio_market_return(self.train_env)
            eval_portfolio_return, eval_market_return, eval_ratio = self._calculate_portfolio_market_return(self.eval_env)
            test_portfolio_return, test_market_return, test_ratio = self._calculate_portfolio_market_return(self.test_env)


            # Enregistrer les métriques dans TensorBoard
            self.logger.record("train/performance/portfolio_return", train_portfolio_return)
            self.logger.record("train/performance/market_return", train_market_return)
            self.logger.record("train/performance/ratio", train_ratio*100)
            self.logger.record("eval/performance/portfolio_return", eval_portfolio_return)
            self.logger.record("eval/performance/market_return", eval_market_return)
            self.logger.record("eval/performance/ratio", eval_ratio*100)
            self.logger.record("test/performance/portfolio_return", test_portfolio_return)
            self.logger.record("test/performance/market_return", test_market_return)
            self.logger.record("test/performance/ratio", test_ratio*100)
            result_str = f"TRAIN: {train_portfolio_return:.0f} / {train_market_return:.0f} = {train_ratio*100:.1f}% | " \
                f"VALIDATION: {eval_portfolio_return:.0f} / {eval_market_return:.0f} = {eval_ratio*100:.1f}% | " \
                f"TEST: {test_portfolio_return:.0f} / {test_market_return:.0f} = {test_ratio*100:.1f}%"
            print(result_str)
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            with open(self.log_dir+'/result.txt', 'at') as f:
                # Rediriger les print vers le fichier
                print(result_str, file=f)
            # Sauvegarder le modèle si le ratio est le meilleur jusqu'à présent
            if isinstance(eval_ratio, (int, float)) and eval_ratio > self.best_ratio:
                self.best_ratio = eval_ratio
                self.model.save(f'{self.model.logger.dir}/best_model_ratio_{eval_ratio*100:.1f}.zip')
                self.model.save(f'{self.model.logger.dir}/best_model.zip')
            print("Current logger values:", self.model.logger.name_to_value)
        return True

    def _calculate_portfolio_market_return(self, env):
        # Calculer le retour du portefeuille ici pour l'environnement spécifié
        obs, info_0 = env.reset()
        done, truncated = False, False
        done = False
        while not truncated and not done:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
            portfolio_valuation = info.get('portfolio_valuation', 0)
        market_valuation = info.get('data_close')/info_0.get('data_open')*info_0.get('portfolio_valuation', 0)
        print(f'truncated: {truncated} | done: {done} | duree: {info_0.get("data_date_close")} -> {info.get("data_date_close")} = {info.get("step")-info_0.get("step")}' \
              f' steps = {int((info.get("step")-info_0.get("step"))/24)} jours')
        if market_valuation != 0:
            ratio = portfolio_valuation / market_valuation
        else:
            ratio = float('inf')
        return portfolio_valuation, market_valuation, ratio

def create_callbacks(eval_envs, log_dir, checkpoint_freq=1000):
    """
    Crée des callbacks pour le suivi de l'entraînement avec évaluation périodique.
    Args:
    eval_envs (gym.Env): Environnement d'évaluation.
    log_dir (str): Répertoire pour enregistrer les logs.
    reward_threshold (float): Seuil de récompense pour arrêter l'entraînement si atteint.
    checkpoint_freq (int): Fréquence pour sauvegarder des checkpoints.
    Returns:
    list: Liste des callbacks.
    """
    os.makedirs(log_dir, exist_ok=True)
    [eval_train_env, eval_val_env, eval_test_env] = eval_envs



    # Callback pour sauvegarder des modèles à des intervalles réguliers
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_freq, save_path=log_dir,
                                             name_prefix='ppo_checkpoint')

    # Callback pour évaluer le modèle périodiquement et sauvegarder le meilleur
    eval_callback = EvalCallback(eval_val_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=5000,
                                 deterministic=True, render=False)
    
    # Callback pour arrêter l'entraînement si la récompense atteint un certain seuil
    tensorboard_callback = TensorboardCallback(eval_train_env, eval_val_env, eval_test_env, log_dir)
    return [checkpoint_callback, eval_callback, tensorboard_callback]

# Function to train the PPO model
def train_ppo(env, eval_env, log_dir, total_timesteps=1000, n_steps=2048, batch_size=64, n_epochs=10, model='CNN'):
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
    Returns:
    stable_baselines3.PPO: Le modèle PPO entraîné.
    """
    # Paramètres spécifiques du modèle
    if model == 'CNN':
        model = CNN_1D_FeatureExtractor
    elif model == 'LSTM':
        model = CNN_LSTM_FeatureExtractor
    elif model == 'Attention':
        model = GTrXL_FeatureExtractor
    else:
        assert 0, 'model does not exist'

    policy_kwargs = dict(
        features_extractor_class=model  # Choisir votre extracteur de features
    )

    # Crée les callbacks
    callbacks = create_callbacks(eval_env, log_dir)

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
    print("Ready to train the model")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    # model.learn(total_timesteps=total_timesteps)
    print("Model trained!")
    return model
