import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from GTrXL import StableTransformerXL

class CNN_1D_FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super(CNN_1D_FeatureExtractor, self).__init__(observation_space, features_dim)

        sequence_length, input_dim = observation_space.shape
        
        # Assume input shape is (sequence_length, feature_dim)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=6, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=6, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )

        # Calculate the size of the output from the CNN
        with th.no_grad():
            sample_input = th.zeros((1, input_dim, sequence_length))  # batch_size, channels, sequence_length
            n_flatten = self.cnn(sample_input).shape[1]

        # Linear layer to obtain the final feature vector
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Swap dimensions to match the expected input for Conv1d: (batch_size, channels, sequence_length)
        observations = observations.permute(0, 2, 1)
        cnn_output = self.cnn(observations)
        return self.linear(cnn_output)


class GTrXL_FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, n_transformer_layers: int=4, n_attn_heads: int=4, features_dim: int = 128):
        super(GTrXL_FeatureExtractor, self).__init__(observation_space, features_dim)

        *seq_len, state_dim = observation_space.shape
        print("Sequence Length (seq_len):", seq_len)
        print("State Dimension (state_dim):", state_dim)
        
        # Assume input shape is (sequence_length, feature_dim)
        self.cnn1d = nn.Sequential(
            nn.Conv1d(in_channels=state_dim, out_channels=64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=features_dim, kernel_size=4, stride=1),
            nn.ReLU()
        )

        # NOTE - I/P Shape : [seq_len, batch_size, state_dim]
        self.transformer = StableTransformerXL(
            d_input=features_dim,
            n_layers=n_transformer_layers,
            n_heads=n_attn_heads,
            d_head_inner=64,  # Exemple de dimension interne pour chaque tête
            d_ff_inner=128,  # Exemple de dimension interne pour les couches feed-forward
            dropout=0.1,  # Taux de dropout
            dropouta=0.1,  # Taux de dropout pour l'attention
            mem_len=150  # Longueur de la mémoire conservée pour les séquences précédentes
        )
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Adapter les dimensions pour CNN1D [batch_size, channels, seq_len]
        if len(observations.shape) > 3:
            observations = observations.squeeze()
        observations = observations.permute(0, 2, 1)
        cnn_output = self.cnn1d(observations)

        # Revenir à la dimension attendue par le transformateur [seq_len, batch_size, new_dim]
        trans_input = cnn_output.permute(2, 0, 1)
        
        trans_state = self.transformer(trans_input)
        trans_state, _ = trans_state['logits'], trans_state['memory']
        return trans_state[-1,...]

class CNN_LSTM_FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128, lstm_hidden_dim: int = 64, lstm_layers: int = 2):
        super(CNN_LSTM_FeatureExtractor, self).__init__(observation_space, features_dim)

        *sequence_length, input_dim = observation_space.shape

        # CNN 1D layers
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # LSTM layer
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)

        # Fully connected layer to match the output dimensions
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Permute dimensions to match Conv1D input
        observations = observations.permute(0, 2, 1)
        cnn_output = self.cnn(observations)

        # Permute back for LSTM
        cnn_output = cnn_output.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(cnn_output)
        
        # Use the last output of LSTM
        lstm_out = lstm_out[:, -1, :]

        # Final fully connected layer
        return self.fc(lstm_out)
