import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_hpim as hpim
import os
import logging
from tests.logging_func import setup_logging, cleanup_logging

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

class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up logging once for the entire test class
        setup_logging(os.path.basename(__file__))

#    @classmethod
#    def tearDownClass(cls):
#        # Clean up logging after all tests are done
#        cleanup_logging()

    def test_weights_and_biases(self):
        """Test for mismatches in weights and biases between original and optimized models."""
        logging.info("Running test_weights_and_biases")

        # Create models
        original_model = BaseModel()
        opt_model = hpim.optimize(model=original_model, mode='layers')

        # Test weights and biases
        mse_list = []
        for orig_layer, opt_layer in zip(original_model.named_modules(), opt_model.named_modules()):
            orig_name, orig_module = orig_layer
            opt_name, opt_module = opt_layer

            if hasattr(orig_module, 'weight') and hasattr(opt_module, 'weight'):
                orig_weights = orig_module.weight.data
                opt_weights = opt_module.weight.data
                if not torch.allclose(orig_weights, opt_weights, atol=1e-2):
                    mse_weights = F.mse_loss(orig_weights, opt_weights)
                    mse_list.append((orig_name + ' weights', mse_weights.item()))
                    logging.warning(f"Mismatch in {orig_name} weights: MSE = {mse_weights.item():.6f}")

            if hasattr(orig_module, 'bias') and hasattr(opt_module, 'bias'):
                orig_bias = orig_module.bias.data
                opt_bias = opt_module.bias.data
                if not torch.allclose(orig_bias, opt_bias, atol=1e-2):
                    mse_bias = F.mse_loss(orig_bias, opt_bias)
                    mse_list.append((orig_name + ' bias', mse_bias.item()))
                    logging.warning(f"Mismatch in {orig_name} bias: MSE = {mse_bias.item():.6f}")

        if not mse_list:
            logging.info("No mismatch in weights and biases.")
        else:
            logging.warning("Mismatch in weights and biases (MSE values):")
            for name, mse in mse_list:
                logging.warning(f"{name}: MSE = {mse:.6f}")

    def test_model_output(self):
        """Test for mismatches in model outputs between original and optimized models."""
        logging.info("Running test_model_output")

        # Create models
        original_model = BaseModel()
        opt_model = hpim.optimize(model=original_model, mode='layers')

        # Generate input tensor
        input_tensor = torch.rand(size=[1, 100])

        # Test model output
        orig_output = original_model(input_tensor)
        opt_output = opt_model(input_tensor)

        if not torch.allclose(orig_output, opt_output, atol=1e-2):
            mse_output = F.mse_loss(orig_output, opt_output)
            logging.warning(f"Mismatch in model output: MSE = {mse_output.item():.6f}")
        else:
            logging.info("No mismatch in model output.")

#    def test_all(self):
#        """Run all model-related tests."""
#        logging.info("Running test_all")
#
#        # Create models
#        original_model = BaseModel()
#        opt_model = hpim.optimize(model=original_model, layers=['linear', 'relu'])
#
#        # Test weights and biases
#        self.test_weights_and_biases()
#
#        # Test model output
#        self.test_model_output()

if __name__ == "__main__":
    unittest.main()

