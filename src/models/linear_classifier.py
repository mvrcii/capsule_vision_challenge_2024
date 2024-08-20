from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(LinearClassifier, self).__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
