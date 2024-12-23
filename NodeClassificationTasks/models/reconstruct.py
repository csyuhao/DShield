import torch
import torch.nn as nn
import torch.optim as optim


class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2 * input_size // 3),
            nn.ReLU(True),
            nn.Linear(2 * input_size // 3, input_size // 3),
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(input_size // 3, 2 * input_size // 3),
            nn.ReLU(True),
            nn.Linear(2 * input_size // 3, input_size),
            nn.Sigmoid()  # Use Sigmoid if the input data is normalized between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class MLPAE(nn.Module):

    def __init__(self, ori_x, trigger, device, epochs):
        super(MLPAE, self).__init__()
        self.device = device
        self.model = Autoencoder(len(ori_x[0])).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epochs = epochs
        self.ori_x = ori_x
        self.trigger = trigger

    def fit(self):
        for epoch in range(self.epochs):
            output = self.model(self.ori_x)
            loss = self.criterion(output, self.ori_x)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def inference(self, input):
        self.model.eval()
        reconstruction_errors = []
        with torch.no_grad():
            for sample in input:
                reconstructed = self.model(sample)
                loss = self.criterion(reconstructed, sample)
                reconstruction_errors.append(loss)

        # Convert the list of tensors to a single tensor
        reconstruction_errors_tensor = torch.stack(reconstruction_errors)
        return reconstruction_errors_tensor
