import d3rlpy
import torch
import torch.nn as nn
import dataclasses

class LSTMEncoder(nn.Module):
    def __init__(self, observation_shape, feature_size):
        super().__init__()
        self.feature_size = feature_size
#        self.fc = nn.Linear(observation_shape[0], 128)
        self.lstm = nn.LSTM(input_size=15, hidden_size=feature_size, batch_first=True)
#        self.rnn_state = (torch.zeros(1, 1, 1).requires_grad_(), torch.zeros(1, 1, 1).requires_grad_())

    def forward(self, inp):
        # Initialize hidden state with zeros
        # sequence length, batch_size, input dimension
        h0 = torch.zeros(1, inp.size(0), 1).requires_grad_()
        c0 = torch.zeros(1, inp.size(0), 1).requires_grad_()

        inp = torch.reshape(inp, [len(inp), 15, 10])
        inp = torch.swapaxes(inp, 1, 2)
#        inp = torch.relu(self.fc(inp))
#        hidden_output, rnn_state_undetached = self.lstm(inp.unsqueeze(dim=0), (h0, c0))
        hidden_output, rnn_state_undetached = self.lstm(inp, (h0, c0))
        # detach?
#        self.rnn_state = (rnn_state_undetached[0].detach(), rnn_state_undetached[1].detach())
        return torch.relu(hidden_output[:,-1,:])


class LSTMEncoderWithAction(nn.Module):
    def __init__(self, observation_shape, action_size, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.fc1 = nn.Linear(observation_shape[0] + action_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, feature_size)
#        self.rnn_state = (torch.zeros(1, 1, 1).requires_grad_(), torch.zeros(1, 1, 1).requires_grad_())

    def forward(self, inp, action):
#        inp = torch.flatten(inp, start_dim=1, end_dim=-1)
        inp = torch.cat([inp, action], dim=1)
        inp = torch.relu(self.fc1(inp))
        inp = torch.relu(self.fc2(inp))
        inp = torch.relu(self.fc3(inp))
        inp = torch.relu(self.fc4(inp))

        return inp


class ACEncoder(nn.Module):
    def __init__(self, observation_shape, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.fc1 = nn.Linear(observation_shape[0], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, feature_size)

    def forward(self, inp):
        inp = torch.relu(self.fc1(inp))
        inp = torch.relu(self.fc2(inp))
        inp = torch.relu(self.fc3(inp))
        outp = torch.tanh(self.fc4(inp))

        return outp


class ACEncoderWithAction(nn.Module):
    def __init__(self, observation_shape, action_size, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.fc1 = nn.Linear(observation_shape[0] + action_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, feature_size)

    def forward(self, inp, action):
        inp = torch.cat([inp, action], dim=1)
        inp = torch.relu(self.fc1(inp))
        inp = torch.relu(self.fc2(inp))
        inp = torch.relu(self.fc3(inp))
        outp = self.fc4(inp)

        return outp


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


@dataclasses.dataclass()
class ACEncoderFactory(d3rlpy.models.EncoderFactory):
    feature_size: int

    def create(self, observation_shape):
        return ACEncoder(observation_shape, feature_size=self.feature_size)

    def create_with_action(self, observation_shape, action_size):
        return ACEncoderWithAction(observation_shape, action_size, self.feature_size)

    @staticmethod
    def get_type() -> str:
        return "ac"
