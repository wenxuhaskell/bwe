import d3rlpy
import torch
import torch.nn as nn
import dataclasses


class LSTMEncoder(nn.Module):
    def __init__(self, observation_shape, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.fc = nn.Linear(observation_shape[0], 128)
        self.lstm = nn.LSTM(128, feature_size, num_layers=1)

    def forward(self, inp):
        # Initialize hidden state with zeros
        # batch_size, sequence length, input dimension
#        h0 = torch.zeros(1, 1, 128).requires_grad_()
        # Initialize cell state
#        c0 = torch.zeros(1, 1, 128).requires_grad_()
        h = torch.relu(self.fc(inp))
#        h, _ = self.lstm(h.unsqueeze(dim=0))
        h, _ = self.lstm(h)
#        h = h[:,-1,:]
        h = torch.relu(h)
        return h


class LSTMEncoderWithAction(nn.Module):
    def __init__(self, observation_shape, action_size, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.fc = nn.Linear(observation_shape[0] + action_size, 128)
        self.lstm = nn.LSTM(128, feature_size, num_layers=1)

    def forward(self, inp, action):
        # Initialize hidden state with zeros
        # batch_size, sequence length, input dimension
#        h0 = torch.zeros(1, 1, 128).requires_grad_()
        # Initialize cell state
#        c0 = torch.zeros(1, 1, 128).requires_grad_()
#        print("inp size: \n")
#        print(inp.size())
#        print("action size: \n")
#        print(action.size())
        h = torch.cat([inp, action], dim=1)
        h = torch.relu(self.fc(h))
#        h, _ = self.lstm(h.unsqueeze(dim=0))
        h, _ = self.lstm(h)
#        h = h[:,-1,:]
        h = torch.relu(h)
        return h


@dataclasses.dataclass()
class LSTMEncoderFactory(d3rlpy.models.EncoderFactory):
    feature_size: int

    def create(self, observation_shape):
        return LSTMEncoder(observation_shape, feature_size=self.feature_size)

    def create_with_action(self, observation_shape, action_size):
        return LSTMEncoderWithAction(observation_shape, action_size, self.feature_size)

    @staticmethod
    def get_type() -> str:
        return "ltsm"

