import d3rlpy
import torch
import torch.nn as nn
import dataclasses

class LSTMEncoder(nn.Module):
    def __init__(self, observation_shape, feature_size):
        super().__init__()
        self.feature_size = feature_size
#        self.fc = nn.Linear(observation_shape[0], 128)
#        self.lstm = nn.LSTM(input_size=15, hidden_size=feature_size, batch_first=True)
        self.lstm_s = nn.LSTM(input_size=15, hidden_size=64, batch_first=True)
        self.lstm_l = nn.LSTM(input_size=15, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, feature_size)

    def forward(self, inp):
        # Initialize hidden state with zeros
        # sequence length, batch_size, input dimension
        h0 = torch.zeros(1, inp.size(0), 64).requires_grad_()
        c0 = torch.zeros(1, inp.size(0), 64).requires_grad_()
        h1 = torch.zeros(1, inp.size(0), 64).requires_grad_()
        c1 = torch.zeros(1, inp.size(0), 64).requires_grad_()

        inp = torch.reshape(inp, [len(inp), 15, 10])
        inp = torch.swapaxes(inp, 1, 2)
        inp_s, inp_l = torch.chunk(inp, 2, dim=1)
        hidden_output_s, _ = self.lstm_s(inp_s, (h0, c0))
        hidden_output_l, _ = self.lstm_l(inp_l, (h1, c1))
        inp = torch.cat((hidden_output_s[:,-1,:], hidden_output_l[:,-1,:]), dim=1)
        inp = self.fc1(torch.relu(inp))
        inp = self.fc2(torch.relu(inp))

        # detach?
#        self.rnn_state = (rnn_state_undetached[0].detach(), rnn_state_undetached[1].detach())
        return torch.relu(inp)


class LSTMEncoderWithAction(nn.Module):
    def __init__(self, observation_shape, action_size, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.fc1 = nn.Linear(observation_shape[0] + action_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 32)
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
