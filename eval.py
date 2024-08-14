from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd

# Function to evaluate the model's performance and track actions
def evaluate_model_with_tracking_by_coin(model, env):
    """
    Évalue le modèle sur l'environnement spécifié et trace les actions et les valeurs du portefeuille
    séparément pour chaque cryptomonnaie.
    Args:
    model (stable_baselines3.PPO): Le modèle à évaluer.
    env (gym.Env): L'environnement sur lequel évaluer le modèle.
    Returns:
    dict: Dictionnaire contenant les DataFrames des résultats de l'évaluation pour chaque coin.
    """
    obs, _ = env.reset()
    done, truncated = False, False
    rewards = []
    portfolio_values = []
    actions = []
    infos = []

    # Boucle de simulation
    while not truncated and not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        actions.append(int(action))
        portfolio_values.append(info['portfolio_valuation'])
        infos.append(info)
    
    # Collecter les résultats par coin
    results = {
        'time': [],
        'portfolio_value': [],
        'actions': [],
        'open': [],
        'close': [],
        'high': [],
        'low': [],
    }
    
    for info in infos:
        results['time'].append(info['date'])
        results['portfolio_value'].append(info['portfolio_valuation'])
        results['actions'].append(int(info['real_position']))
        results['open'].append(info['data_open'])
        results['close'].append(info['data_close'])
        results['high'].append(info['data_high'])
        results['low'].append(info['data_low'])

    # Convertir les résultats en DataFrames
    df_results = pd.DataFrame(results)
    df_results.set_index('time', inplace=True)

    market_return = df_results.iloc[-1]['close']/df_results.iloc[0]['open']-1
    portfolio_return = df_results.iloc[-1]['portfolio_value']/df_results.iloc[0]['portfolio_value']-1
    num_achats = ((df_results['actions'].shift(1) == 0) & (df_results['actions'] == 1)).sum()

    return df_results, {'market_return':market_return*100, 'portfolio_return':portfolio_return*100, 'num_achats':num_achats}


def calculate_performance(performance_records, output_dir):
    """
    Calcule et sauvegarde les performances d'un modèle de trading pour différents ensembles de données (train/test),
    exchanges, et coins.

    Args:
    performance_records (list of dict): Liste de dictionnaires où chaque dictionnaire contient les performances 
                                        pour un ensemble de données, un exchange et un coin spécifiques.
                                        Exemple de structure de dictionnaire :
                                        {
                                            "Set": "Test",
                                            "Exchange": "binance",
                                            "Coin": "BTCUSDT",
                                            "Portfolio Return": 0.05,
                                            "Market Return": 0.03
                                        }
    output_dir (str): Répertoire où les fichiers CSV de performances seront sauvegardés.

    Sauvegarde:
    - `performance_details.csv`: Contient les performances détaillées pour chaque ensemble de données, exchange, et coin.
    - `performance_summary.csv`: Contient les moyennes des performances par ensemble (train/test), exchange, et coin,
                                 ainsi que les moyennes globales pour chaque ensemble.
    """
    # Créer un DataFrame Pandas pour stocker les résultats
    performance_df = pd.DataFrame(performance_records)
    
    # Sélectionner uniquement les colonnes numériques pour le calcul des moyennes
    numeric_cols = performance_df.select_dtypes(include=['number']).columns

    # Conserver les colonnes de regroupement
    group_cols = ['Set', 'Exchange', 'Coin']

    # Calcul des moyennes par ensemble, exchange et coin en utilisant seulement les colonnes numériques
    avg_performance = performance_df[group_cols + list(numeric_cols)].groupby(group_cols).mean().reset_index()

    # Ajouter les moyennes générales pour chaque ensemble
    overall_avg = performance_df.groupby('Set')[numeric_cols].mean().reset_index()
    overall_avg['Exchange'] = "All"
    overall_avg['Coin'] = "All"
    avg_performance = pd.concat([avg_performance, overall_avg], ignore_index=True)
    
    # Calcul des moyennes par exchange (sur tous les coins)
    avg_by_exchange = performance_df.groupby(['Set', 'Exchange'])[numeric_cols].mean().reset_index()
    avg_by_exchange['Coin'] = "All"  # Marquer que c'est une moyenne sur tous les coins
    avg_performance = pd.concat([avg_performance, avg_by_exchange], ignore_index=True)


    # Calcul des moyennes par coin (sur tous les exchanges)
    avg_by_coin = performance_df.groupby(['Set', 'Coin'])[numeric_cols].mean().reset_index()
    avg_by_coin['Exchange'] = "All"  # Marquer que c'est une moyenne sur tous les coins
    avg_performance = pd.concat([avg_performance, avg_by_coin], ignore_index=True)
    
    # Sauvegarder le tableau final en CSV
    performance_df.to_csv(f"{output_dir}/performance_details.csv", index=False)
    avg_performance.to_csv(f"{output_dir}/performance_summary.csv", index=False)


# Plotting the results
def plot_results(data_plot, name):
    """
    Trace les résultats de l'évaluation du modèle.
    Args:
    data (pd.DataFrame): DataFrame contenant les résultats de l'évaluation.
    name (str): Nom de base pour les fichiers de sortie.
    """
    # Assurer que l'index est en datetime
    if not isinstance(data_plot.index, pd.DatetimeIndex):
        data_plot.index = pd.to_datetime(data_plot.index)

    # Préparation des signaux
    data_plot['actions_shifted'] = data_plot['actions'].shift(1)
    buy_signals = data_plot[(data_plot['actions'] == 1) & (data_plot['actions_shifted'] == 0)]
    print(f'Number of buy:{len(buy_signals)}')
    sell_signals = data_plot[(data_plot['actions'] == 0) & (data_plot['actions_shifted'] == 1)]

    # Créer des subplots avec un espacement vertical augmenté
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.15, subplot_titles=('Price', 'Portfolio Value'),
                        row_width=[0.2, 0.7], specs=[[{}], [{"secondary_y": True}]])

    # Ajouter les chandeliers au premier subplot
    fig.add_trace(go.Candlestick(x=data_plot.index,
                                 open=data_plot['open'], high=data_plot['high'],
                                 low=data_plot['low'], close=data_plot['close'],
                                 name='Candlestick'), row=1, col=1)

    # Ajouter les signaux d'achat et de vente
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['close'],
                             mode='markers', name='Buy Signal',
                             marker=dict(color='Green', size=10, symbol='triangle-up')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['close'],
                             mode='markers', name='Sell Signal',
                             marker=dict(color='Red', size=10, symbol='triangle-down')),
                  row=1, col=1)

    # Ajouter la courbe de valeur du portefeuille au second subplot
    fig.add_trace(go.Scatter(x=data_plot.index, y=data_plot['portfolio_value'],
                             mode='lines', name='Portfolio Value',
                             line=dict(color='purple')), row=2, col=1)

    # Mise à jour de la mise en page pour un affichage optimal
    fig.update_layout(title='Interactive Financial Chart with Portfolio Value',
                      xaxis_title='Date', yaxis_title='Price',
                      template='plotly_dark', showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      width=1000, height=800)  # Taille ajustée

    # Ajuster les échelles en y pour chaque subplot
    fig.update_yaxes(range=[min(data_plot['low']), max(data_plot['high'])], row=1, col=1)  # Pour les prix
    fig.update_yaxes(range=[min(data_plot['portfolio_value']), max(data_plot['portfolio_value'])], row=2, col=1)  # Pour la valeur du portefeuille

    # Exporter le graphique en HTML
    fig.write_html(f'{name}.html')

    # Deuxieme plot plus simple
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)

    # Plot Bitcoin Close Prices
    axes[0].plot(data_plot.index, data_plot['close'], label='Bitcoin Close Price', alpha=0.7)
    axes[0].set_title('Bitcoin Close Price')
    axes[0].set_ylabel('Price')
    axes[0].legend(loc='upper left')

    # Mark Buy/Sell Actions on Close Price
    buy_signals = data_plot[(data_plot['actions'] == 1) & (data_plot['actions'].shift(1) == 0)]
    sell_signals = data_plot[(data_plot['actions'] == 0) & (data_plot['actions'].shift(1) == 1)]

    axes[0].scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='Buy Signal', s=100)
    axes[0].scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='Sell Signal', s=100)

    # Plot Portfolio Returns
    axes[1].plot(data_plot.index, data_plot['portfolio_value'], label='Portfolio Value', alpha=0.7, color='green')
    axes[1].set_title('Portfolio Return Over Time')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Portfolio Value')
    axes[1].legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(f'{name}.png')
    plt.close()
