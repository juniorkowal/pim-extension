import torch
import torch.nn as nn
import hpim


class TinyModel(nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = nn.Linear(100, 200)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(200, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
    

class BiggerModel(nn.Module):

    def __init__(self):
        super(BiggerModel, self).__init__()    

        self.tinymodel = TinyModel()
        self.linear1 = nn.Linear(10, 20)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.tinymodel(x)
        x = self.linear1(x)
        x = self.activation(x)
        return x
    

if __name__ == "__main__":
    print(torch.__version__)

    input_tens = torch.rand(size = [100])

    test_model1 = TinyModel()
    test_model2 = BiggerModel()

    print(test_model1(input_tens))

    switched_model = hpim.optimize(model = test_model1, layers=['linear', 'relu'])

    print(switched_model(input_tens))
    for named_module in switched_model.named_children():
        print(named_module)

