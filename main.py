""" Script pour trainer / test un algo PPO de RL sur en environnement de env-trading"""
import os
import datetime
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO

from data import download_and_split_data
from environnement import create_multi_env, create_env
from train import train_ppo
from eval import evaluate_model_with_tracking_by_coin, plot_results, calculate_performance

def extract_coin_and_exchange(filename):
    # Séparer la chaîne par le délimiteur '-'
    parts = filename.split('-')

    # Le premier élément est l'exchange
    exchange = parts[0]

    # Le deuxième élément est la paire de cryptos (par exemple "XRPUSDT")
    coin_pair = parts[1]

    return exchange, coin_pair


def main(args):
    """ Main execution for training / testing"""
    log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(f'{log_dir}/output/', exist_ok=True)

    # Dataloading
    download_and_split_data()

    # Créez les environnements pour chaque dataset
    train_env = create_multi_env(dataset_dir='data/train/*.pkl', log_dir=log_dir,  state_window_size=args.state_window_size,
                                 max_episode_duration=args.max_duration_train, short=args.short)
    val_env = create_multi_env(dataset_dir='data/val/*.pkl', log_dir=log_dir, state_window_size=args.state_window_size, short=args.short)

    # Entraîner le modèle PPO avec l'ensemble d'entraînement et évaluer sur l'ensemble de validation/test
    if os.path.exists(args.model_path):
        model = PPO.load(args.model_path)
    else:
        model = train_ppo(train_env, val_env, log_dir=log_dir, total_timesteps=args.total_timesteps, n_steps=args.n_steps, 
                        batch_size=args.batch_size, n_epochs=args.n_epochs)

    # Évaluation sur l'ensemble de test, séparé par coin
    performance_records = []
    for dataset in ['test', 'train']:
        for file_name in os.listdir(f'data/{dataset}'):
            if file_name.endswith(".pkl"):
                exchange, coin_pair = extract_coin_and_exchange(file_name)
                print(f'Testing: {exchange} with {coin_pair} on {dataset} set')
                test_envs = create_env(dataset_path=f'data/{dataset}/{file_name}', log_dir=log_dir, state_window_size=args.state_window_size, short=args.short)
                df_results, stats = evaluate_model_with_tracking_by_coin(model, test_envs)
                plot_results(df_results, f"{log_dir}/output/{dataset}_{exchange}_{coin_pair}")
                test_envs.unwrapped.save_for_render(dir=f"{log_dir}/output/{dataset}_{exchange}_{coin_pair}")
                # Ajouter les performances au DataFrame
                performance_records.append({
                    "Set": dataset,
                    "Exchange": exchange,
                    "Coin": coin_pair,
                    "Portfolio Return": stats['portfolio_return'],
                    "Market Return": stats['market_return'],
                    "Number Buys": stats['num_achats'],
                })
    calculate_performance(performance_records, log_dir)
                
if __name__ == "__main__":
    # Si pas fixé, les résultats ne sont pas constant. #TODO find which is problematic
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--state_window_size", type=int, default=4*24, help="Window size for state representation.")
    parser.add_argument("--max_duration_train", type=int, default=30*24, help="Maximum duration for training episodes.")
    parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps for the optimizer.")
    parser.add_argument("--n_epochs", type=int, default=8, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--total_timesteps", type=int, default=100_000, help="Total timesteps for training.")
    parser.add_argument("--short", action="store_true", help="Enable short selling.")
    parser.add_argument("--model_path", type=str, default="logs/20240814-165041/best_model.zip", help="Path to save the trained model.")
    main(parser.parse_args())
