import itertools
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime

from BweReward import process_feature_qoev3
from BweUtils import load_multiple_files, load_train_data_from_file

# Convert data to torch tensors
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


class MyNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        self.fc2 = nn.Linear(256, 256)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
        self.fc3 = nn.Linear(256, 256)
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity="relu")
        self.fc4 = nn.Linear(256, output_dim)
        nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity="tanh")

    def forward(self, inp):
        inp = torch.relu(self.fc1(inp))
        inp = torch.relu(self.fc2(inp))
        inp = torch.relu(self.fc3(inp))
        oup = torch.tanh(self.fc4(inp))

        return oup

def calc_qoe(ve, ae) -> float:
    qoe = 0.0
    if not np.isnan(ve):
        if not np.isnan(ae):
            qoe = (ve+ae) * 0.5
        else:
            qoe = ve
    elif not np.isnan(ae):
        qoe = ae

    return qoe


def save_model(model, batch_size, x, y):
    x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "super_resolution.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_dir = f'QOE_{ts}'

    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idir", type=str, default="data")
    parser.add_argument("-b", "--batchsize", type=int, default=32)
    parser.add_argument("-m", "--maxfiles", type=int, default=3)
    parser.add_argument("-e", "--numepochs", type=int, default=100)
    args = parser.parse_args()

    batch_size = args.batchsize
    # load the list of log files under the given directory
    # iterate over files in that directory
    data_files = load_multiple_files(args.idir, args.maxfiles, random_choice=True)
    observations = []
    videos = []
    audios = []
    for filename in data_files:
        result = load_train_data_from_file(filename)
        bw_obs, bw_pred, reward, term, video, audio, capacity, lossrate = result
        # feature reduction
        bw_obs = process_feature_qoev3(bw_obs)
        observations.append(bw_obs)
        videos.append(video)
        audios.append(audio)

    observations = np.concatenate(observations)
    videos = np.concatenate(videos)
    audios = np.concatenate(audios)
    # calculate QoE
    qoes = np.array([calc_qoe(v,a) for (v, a) in zip(videos, audios)])

    # scaling
    obs_scaler = StandardScaler()
    observations_scaled = obs_scaler.fit_transform(observations)
    qoe_scaler = MinMaxScaler()
    qoes_scaled = qoe_scaler.fit_transform(qoes.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(observations_scaled, qoes_scaled, test_size=0.2, random_state=26)
    # Instantiate training and test data
    train_data = Data(X_train, y_train)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    test_data = Data(X_test, y_test)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    # Check it's working
    for batch, (X, y) in enumerate(train_dataloader):
        print(f"Batch: {batch+1}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        break

    print(observations.shape)

    # training
    model = MyNN(observations.shape[1], 1)
    print(model)

    learning_rate = 0.1

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = args.numepochs
    loss_values = []

    for epoch in range(num_epochs):
        for X, y in train_dataloader:
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            pred = model(X)
            loss = loss_fn(pred, y)
            print(f"Epoch {epoch}/{num_epochs}, loss {loss.item()}.")
            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()

    print("Training Complete")


    """
    Training Complete
    """

    steps = range(len(loss_values))

    plt.plot(steps, np.array(loss_values))
    plt.title("Step-wise Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    """
    We're not training so we don't need to calculate the gradients for our outputs
    """
    y_pred = []
    y_test = []
    with torch.no_grad():
        for X, y in test_dataloader:
            predicted = model(X)
            y_pred.append(predicted)
            y_test.append(y)


    y_pred = list(itertools.chain(*y_pred))
    y_test = list(itertools.chain(*y_test))

    x = range(len(y_pred))

#    plt.plot(1,2,1)
    plt.plot(x, y_pred, label="model estimate scaled")
    plt.plot(x, y_test, label="existing estimate scaled")
    plt.legend()
    plt.xlabel('step')
    plt.ylabel('QoE')
    plt.show()

    predicted_rev = qoe_scaler.inverse_transform(y_pred)
    test_rev = qoe_scaler.inverse_transform(y_test)

#    plt.subplot(1,2,2)
    plt.plot(x, predicted_rev, label="model estimate")
    plt.plot(x, test_rev, label="existing estimate")
    plt.legend()
    plt.xlabel('step')
    plt.ylabel('QoE')

    plt.show()

if __name__ == "__main__":

    main()

