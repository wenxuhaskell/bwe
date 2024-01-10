import d3rlpy
import torch
import torch.nn as nn
import dataclasses


class LSTMEncoder(nn.Module):
    def __init__(self, observation_shape, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.fc = nn.Linear(observation_shape[0], 128)
        self.lstm = nn.LSTM(128, feature_size)
        self.rnn_state = (torch.zeros(1, 1, 1).requires_grad_(), torch.zeros(1, 1, 1).requires_grad_())

    def forward(self, inp, h=None, c=None):
        # Initialize hidden state with zeros
        # batch_size, sequence length, input dimension
        inp = torch.relu(self.fc(inp))
        hidden_output, rnn_state_undetached = self.lstm(inp.unsqueeze(dim=0), self.rnn_state)
        # detach?
        self.rnn_state = (rnn_state_undetached[0].detach(), rnn_state_undetached[1].detach())
        return torch.relu(hidden_output[:, -1, :])


class LSTMEncoderWithAction(nn.Module):
    def __init__(self, observation_shape, action_size, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.fc = nn.Linear(observation_shape[0] + action_size, 128)
        self.lstm = nn.LSTM(128, feature_size)
        self.rnn_state = (torch.zeros(1, 1, 1).requires_grad_(), torch.zeros(1, 1, 1).requires_grad_())

    def forward(self, inp, action, h=None, c=None):
        # Initialize hidden state with zeros
        # batch_size, sequence length, input dimension
        inp = torch.cat([inp, action], dim=1)
        inp = torch.relu(self.fc(inp))

        hidden_output, rnn_state_undetached = self.lstm(inp.unsqueeze(dim=0), self.rnn_state)
        # detach?
        self.rnn_state = (rnn_state_undetached[0].detach(), rnn_state_undetached[1].detach())
        return torch.relu(hidden_output[:, -1, :])


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
