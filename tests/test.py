import torch
import torch.nn as nn
import hpim
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

        self.linear1 = nn.Linear(100, 200)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(200, 10)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

class NestedModel(nn.Module):
    def __init__(self):
        super(NestedModel, self).__init__()

        self.tinymodel = BaseModel()
        self.linear1 = nn.Linear(10, 20)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.tinymodel(x)
        x = self.linear1(x)
        x = self.activation(x)
        return x


def test_weights_and_biases(original_model: nn.Module, opt_model: nn.Module):
    mse_list = []
    
    for orig_layer, opt_layer in zip(original_model.named_modules(), opt_model.named_modules()):
        orig_name, orig_module = orig_layer
        opt_name, opt_module = opt_layer

        if hasattr(orig_module, 'weight') and hasattr(opt_module, 'weight'):
            orig_weights = orig_module.weight.data
            opt_weights = opt_module.weight.data
            if not torch.allclose(orig_weights, opt_weights):
                mse_weights = F.mse_loss(orig_weights, opt_weights)
                mse_list.append((orig_name + ' weights', mse_weights.item()))

        if hasattr(orig_module, 'bias') and hasattr(opt_module, 'bias'):
            orig_bias = orig_module.bias.data
            opt_bias = opt_module.bias.data
            if not torch.allclose(orig_bias, opt_bias):
                mse_bias = F.mse_loss(orig_bias, opt_bias)
                mse_list.append((orig_name + ' bias', mse_bias.item()))

    return mse_list

def test_model_output(original_model: nn.Module, 
                      opt_model: nn.Module, 
                      input_tensor: torch.Tensor):
    
    orig_output = original_model(input_tensor)
    opt_output = opt_model(input_tensor)
    
    if not torch.allclose(orig_output, opt_output):
        mse_output = F.mse_loss(orig_output, opt_output)
        return mse_output.item()
    return 0.0
    
def test_all(original_model: nn.Module, opt_model: nn.Module):
    input_tens = torch.rand(size=[1, 100])
    mse_weights_and_biases = test_weights_and_biases(original_model, opt_model)
    if mse_weights_and_biases:
        print("Mismatch in Weights and Biases (MSE values):")
        for name, mse in mse_weights_and_biases:
            print(f"{name}: MSE = {mse}")
    else:
        print("No mismatch in Weights and Biases.")

    mse_output = test_model_output(original_model, opt_model, input_tens)
    if mse_output > 0:
        print(f"Mismatch in output (MSE): {mse_output}")
    else:
        print("No mismatch in output.")


if __name__ == "__main__":
    import copy
    print(torch.__version__)
    torch.manual_seed(0)

    test_model = BaseModel()
    test_model_copy = copy.deepcopy(test_model)
    opt_model = hpim.optimize(model=test_model, layers=['linear', 'relu'])

    with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=True) as profiler:
        test_all(test_model_copy, opt_model)
        print(hpim.ops.mm(torch.rand(2,2), torch.rand(2,2)))
#        print(opt_model(torch.rand(size=[1, 100])).shape)
    print(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
