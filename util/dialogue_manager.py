from transformers import DistilBertForSequenceClassification
from torch import nn

class DialogueManagerModel(nn.Module):
    DEFAULT_MODEL = "distilbert-base-uncased"

    def __init__(self, n_classes, model_name=None, device='cpu'):
        super().__init__()
        if model_name is None:
            self.model = DistilBertForSequenceClassification.from_pretrained(self.DEFAULT_MODEL)
        else:
            raise NotImplementedError()
        self.model.to(device)
        self.n_classes = n_classes
        self.freeze_layers()
        self.model.classifier = nn.Linear(self.model.classifier.in_features, self.n_classes,
                                          device=device)

        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, X):
        return self.model(X)