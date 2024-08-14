import os
import datetime
import pandas as pd
from gym_trading_env.downloader import download
from ta.volatility import BollingerBands

# Fonction pour charger les données du marché
def download_and_split_data(
    n_days=7*365,
    timeframe="1h",
    symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT"],
    exchanges=["binance", "bitfinex2", "huobi"],
    output_dir="data"
):
    """
    Charge ou télécharge les données de marché depuis plusieurs échanges pour plusieurs cryptos.
    Args:
    n_days (int): Nombre de jours de données à télécharger.
    timeframe (str): Intervalle de temps des données (ex. "1h" pour une heure).
    symbols (list): Liste des paires de cryptomonnaies à télécharger (ex. ["BTC/USDT", "ETH/USDT"]).
    exchanges (list): Liste des échanges à partir desquels télécharger les données (ex. ["binance", "bitfinex2"]).
    output_dir (str): Répertoire où enregistrer les données téléchargées.
    Returns:
    pd.DataFrame: DataFrame contenant les données de marché.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        download(
            exchange_names=exchanges,
            symbols=symbols,
            timeframe=timeframe,
            dir=output_dir,
            since=datetime.datetime.now() - datetime.timedelta(days=n_days),
            until=datetime.datetime.now(),
        )
 
    if not os.path.exists(output_dir+'/train/') or not os.path.exists(output_dir+'/val/') or not os.path.exists(output_dir+'/test/'):
        process_data_splits(data_dir="data")

# Fonction principale pour charger, splitter et sauvegarder les données
def process_data_splits(data_dir="data"):
    """
    Charge les données depuis data_dir, les split en train/val/test et les sauvegarde dans les répertoires appropriés.
    Args:
    data_dir (str): Répertoire contenant les fichiers de données.
    """
    # Parcourir tous les fichiers de données
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pkl"):
            file_path = os.path.join(data_dir, file_name)
            print(f"Processing {file_path}...")

            # Charger les données
            data = pd.read_pickle(file_path)

            # Split les données
            train_data, val_data, test_data = split_data(data)

            # Extraire le nom du fichier sans l'extension
            base_name = os.path.splitext(file_name)[0]

            # Sauvegarder les données splitées
            save_data_splits(train_data, val_data, test_data, base_dir=data_dir, filename=base_name)


# Fonction pour diviser les données en ensembles d'entraînement, de validation et de test
def split_data(data):
    """
    Divise les données en ensembles d'entraînement, de validation, et de test.
    Args:
    data (pd.DataFrame): DataFrame contenant les données de marché brutes.
    Returns:
    tuple: (train_data, val_data, test_data)
    """
    # Définir les dates de coupure
    today = datetime.datetime.now()
    val_start_date = today - datetime.timedelta(days=365)
    test_start_date = today - datetime.timedelta(days=6*30)

    # Filtrer les données
    train_data = data[data.index < val_start_date]
    val_data = data[(data.index >= val_start_date) & (data.index < test_start_date)]
    test_data = data[(data.index >= test_start_date)]

    return train_data, val_data, test_data

# Fonction pour sauvegarder les données dans les répertoires appropriés
def save_data_splits(train_data, val_data, test_data, base_dir="data_splits", filename=""):
    """
    Sauvegarde les ensembles d'entraînement, de validation et de test dans des répertoires séparés.
    Args:
    train_data (pd.DataFrame): Données d'entraînement.
    val_data (pd.DataFrame): Données de validation.
    test_data (pd.DataFrame): Données de test.
    base_dir (str): Répertoire de base pour sauvegarder les données.
    """
    os.makedirs(base_dir, exist_ok=True)

    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_data.to_pickle(os.path.join(train_dir, f"{filename}.pkl"))
    val_data.to_pickle(os.path.join(val_dir, f"{filename}.pkl"))
    test_data.to_pickle(os.path.join(test_dir, f"{filename}.pkl"))

# Calcul des indicateurs techniques et normalisation des données
def preprocess_data(data):
    """
    Prétraitement des données du marché en calculant divers indicateurs techniques et en normalisant les données.
    Args:
    data (pd.DataFrame): DataFrame contenant les données de marché brutes.
    Returns:
    pd.DataFrame: DataFrame avec des indicateurs techniques ajoutés et des données normalisées.
    """

    # Calculate SMA & EMA at different periods
    periods = [7, 20, 40, 60, 100, 200]
    for period in periods:
        data[f'feature_SMA_{period}'] = data['close'].rolling(window=period).mean()
        data[f'feature_EMA_{period}'] = data['close'].ewm(span=period, adjust=False).mean()

    # Calculate RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['feature_RSI'] = 100 - (100 / (1 + rs))

    # Calculate MACD and Signal
    ema_12 = data['close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['close'].ewm(span=26, adjust=False).mean()
    data['feature_MACD'] = ema_12 - ema_26
    data['feature_Signal'] = data['feature_MACD'].ewm(span=9, adjust=False).mean()

    # Calculate support and resistance levels
    data['feature_Close'] = data['close']
    data['feature_Support'] = data['low'].rolling(window=20).min()
    data['feature_Resistance'] = data['high'].rolling(window=20).max()
    data['feature_Prev_High'] = data['high'].shift(1).rolling(window=20).max()
    data['feature_Prev_Low'] = data['low'].shift(1).rolling(window=20).min()
    data['feature_Distance_High'] = data['high'] - data['feature_Prev_High']
    data['feature_Distance_Low'] = data['low'] - data['feature_Prev_Low']
    data['feature_Candle_Dynamics'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['feature_Distance_Open_Close'] = (data['open'] - data['close']) / (data['high'] - data['low'])
    data['feature_Distance_High_Close'] = (data['high'] - data['close']) / (data['high'] - data['low'])
    data['feature_Distance_Low_Close'] = (data['low'] - data['close']) / (data['high'] - data['low'])

    data['feature_ADX'] = calculate_adx(data)

    data['feature_MFI'] = calculate_mfi(data)

    data['feature_VWAP'] = calculate_rolling_vwap(data)

    # Calculate BB
    indicator_bb = BollingerBands(close=data["close"], window=20, window_dev=2)
    data['feature_bb_bbm'] = indicator_bb.bollinger_mavg()
    data['feature_bb_bbh'] = indicator_bb.bollinger_hband()
    data['feature_bb_bbl'] = indicator_bb.bollinger_lband()

    # Add Parabolic Stop and Reverse (Parabolic SAR) indicator
    data['feature_psar'] = calculate_psar(high=data["high"], low=data["low"], close=data["close"], step=0.02, max_step=2)

    # Normalization using the mean and std of Close
    close_mean = data['close'].mean()
    close_std = data['close'].std()

    # Columns to be normalized
    columns_to_normalize = ['feature_Close', 'feature_SMA_7', 'feature_EMA_7', 'feature_SMA_20', 'feature_EMA_20', 'feature_SMA_40', 'feature_EMA_40', 'feature_SMA_60', 'feature_EMA_60',
                            'feature_SMA_100', 'feature_EMA_100', 'feature_SMA_200', 'feature_EMA_200', 'feature_Support', 'feature_Resistance',
                            'feature_Prev_High', 'feature_Prev_Low', 'feature_VWAP', 'feature_MACD', 'feature_bb_bbm', 'feature_bb_bbh', 'feature_bb_bbl', 'feature_psar']

    # Apply normalization to the specified columns
    data[columns_to_normalize] = data[columns_to_normalize].apply(lambda x: (x - close_mean) / close_std)

    # Normalization of remaining indicators
    data['feature_RSI'] = (data['feature_RSI'] - 50) / 50
    data['feature_Distance_High'] = (data['feature_Distance_High']) / close_std
    data['feature_Distance_Low'] = (data['feature_Distance_Low']) / close_std
    data['feature_MFI'] = (data['feature_MFI'] - 50) / 100  # Center MFI around its midpoint
    data['feature_ADX'] = (data['feature_ADX'] - 50) / 100  # Center ADX around its midpoint
    data['feature_Volume'] = (data['volume'] - data['volume'].mean()) / data['volume'].std()
    data['feature_Signal'] = (data['feature_Signal'] ) / close_std  # Center ADX around its midpoint

    data.dropna(inplace=True)
    return data

# Calculate PSAR
def calculate_psar(high, low, close, step=0.02, max_step=0.2):
    """
    Calcule le Parabolic SAR (PSAR).
    Args:
    high (pd.Series): Série des prix hauts.
    low (pd.Series): Série des prix bas.
    close (pd.Series): Série des prix de clôture.
    step (float): Incrément du facteur d'accélération.
    max_step (float): Facteur d'accélération maximal.
    Returns:
    pd.Series: Série du PSAR.
    """
    psar = close.copy()
    psar.iloc[0] = low.iloc[0]
    bull = True
    af = step
    ep = high.iloc[0]
    for i in range(1, len(psar)):
        prev_psar = psar.iloc[i-1]
        if bull:
            psar.iloc[i] = prev_psar + af * (ep - prev_psar)
            if high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + step, max_step)
            if low.iloc[i] < psar.iloc[i]:
                bull = False
                psar.iloc[i] = ep
                ep = low.iloc[i]
                af = step
        else:
            psar.iloc[i] = prev_psar + af * (ep - prev_psar)
            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + step, max_step)
            if high.iloc[i] > psar.iloc[i]:
                bull = True
                psar.iloc[i] = ep
                ep = high.iloc[i]
                af = step

    return psar

# Calculate ADX
def calculate_adx(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    plus_dm = high.diff().where((high.diff() > low.diff()) & (high.diff() > 0), 0.0)
    minus_dm = low.diff().where((low.diff() > high.diff()) & (low.diff() > 0), 0.0)
    tr = pd.concat([high - low, high - close.shift(), close.shift() - low], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).abs() * 100
    adx = dx.rolling(window=period).mean()
    return adx

# Calculate MFI
def calculate_mfi(df, period=14):
    # Calculate typical price
    typical_price = (df['high'] + df['low'] + df['close']) / 3

    # Calculate raw money flow
    money_flow = typical_price * df['volume']

    # Calculate positive and negative money flows
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    # Calculate sums of positive and negative money flows over the period
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()

    # Calculate money flow ratio
    mfr = positive_mf / negative_mf

    # Calculate MFI
    mfi = 100 - (100 / (1 + mfr))

    return mfi

# Calculate VWAP
def calculate_rolling_vwap(df, window=20):
    # Calculate typical price
    typical_price = (df['high'] + df['low'] + df['close']) / 3

    # Calculate rolling VWAP
    vwap = (typical_price * df['volume']).rolling(window).sum() / df['volume'].rolling(window).sum()

    return vwap
