import torch.nn as nn
import torch.nn.functional as F


class MILExpert(nn.Module):
    def __init__(self, model: nn.Module, model_dim: int, drop_p=0.1):
        super(MILExpert, self).__init__()
        self.model = model
        self.drop = nn.Dropout(drop_p)
        # self.bn = nn.BatchNorm1d(model_dim)

    def forward(self, x, mask=None):
        """
        Expects input of shape:
          x: B, N, D
          mask: B, N, 1
        Where:
          B: Batch size
          N: Number of patch features
          D: The dimension of the selected expert backbone

        Example:
        We chose the ResNet50 backbone, which has a dimension of 2048.
        The expected input shape would be:
          B, N, 2048
        """
        x = self.drop(x)
        return F.softmax(self.model(x, mask), dim=1)


class Expert(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 512,
        drop_p: float = 0.1,
    ):
        super(Expert, self).__init__()
        self.drop1 = nn.Dropout(drop_p)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.relu = nn.ReLU()
        self.drop2 = nn.Dropout(drop_p)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        Expects input of shape:
          x: B, D
        Where:
          B: Batch size
          D: The dimension of the selected expert backbone

        Example:
        We chose the ResNet50 backbone, which has a dimension of 2048.
        The expected input shape would be:
          B, 2048
        """
        # B, D = x.shape
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        return x
