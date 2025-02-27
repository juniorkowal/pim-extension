import unittest
import logging
import torch
import random
import os
import hpim  # Import your custom operator
from tests.logging_func import setup_logging, cleanup_logging  # Import the logging setup function

class TestMM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up logging once for the entire test class
        setup_logging(os.path.basename(__file__))

#    @classmethod
#    def tearDownClass(cls):
#        # Clean up logging after all tests are done
#        cleanup_logging()

    def test_mm_simple(self):
        """Test matrix multiplication with simple 2x2 matrices."""
        logging.info("Running test_mm_simple")
        
        mat = torch.randn(2, 2)
        
        result_hpim = hpim.ops.mm(mat, mat)
        result_torch = torch.mm(mat, mat)
        
        # Check for NaNs in the result
        if torch.isnan(result_hpim).any():
            logging.warning("NaN detected in result of test_mm_simple (hpim).")
        
        # Compare results with atol=1e-2
        if not torch.allclose(result_hpim, result_torch, atol=1e-2):
            # Calculate Mean Squared Error (MSE)
            mse = torch.mean((result_hpim - result_torch) ** 2).item()
            logging.warning(f"test_mm_simple mismatch: MSE = {mse:.6f}")
        else:
            logging.info("test_mm_simple: hpim.mm and torch.mm results are the same.")
        
        logging.info("test_mm_simple completed successfully")

    def test_mm_large(self):
        """Test matrix multiplication with large matrices."""
        logging.info("Running test_mm_large")
        
        n, m, o = 300, 129, 317  # Matrix dimensions
        mat1 = torch.randn(n, m)
        mat2 = torch.randn(m, o)
        
        result_hpim = hpim.ops.mm(mat1, mat2)
        result_torch = torch.mm(mat1, mat2)
        
        # Check for NaNs in the result
        if torch.isnan(result_hpim).any():
            logging.warning("NaN detected in result of test_mm_large (hpim).")
        
        # Compare results with atol=1e-2
        if not torch.allclose(result_hpim, result_torch, atol=1e-2):
            # Calculate Mean Squared Error (MSE)
            mse = torch.mean((result_hpim - result_torch) ** 2).item()
            logging.warning(f"test_mm_large mismatch: MSE = {mse:.6f}")
        else:
            logging.info("test_mm_large: hpim.mm and torch.mm results are the same.")
        
        logging.info("test_mm_large completed successfully")

    def test_mm_loop(self):
        """Test matrix multiplication with random matrices in a loop."""
        logging.info("Running test_mm_loop")
        
        num_iter = 10  # Number of iterations
        mat_range = 300  # Maximum matrix size
        
        for i in range(num_iter):
            rows_mat1 = random.randint(1, mat_range)
            cols_mat1 = random.randint(1, mat_range)
            rows_mat2 = cols_mat1
            cols_mat2 = random.randint(1, mat_range)
            
            logging.info(f"Iteration {i+1}: mat1 size: ({rows_mat1}, {cols_mat1}), mat2 size: ({rows_mat2}, {cols_mat2})")

            mat1 = torch.randn(rows_mat1, cols_mat1)
            mat2 = torch.randn(rows_mat2, cols_mat2)

            result_torch = torch.mm(mat1, mat2)
            result_hpim = hpim.ops.mm(mat1, mat2)

            # Check for NaNs in the result
            if torch.isnan(result_hpim).any():
                logging.warning(f"NaN detected in result at iteration {i+1}")

            # Compare results with atol=1e-2
            if not torch.allclose(result_hpim, result_torch, atol=1e-2):
                # Calculate Mean Squared Error (MSE)
                mse = torch.mean((result_hpim - result_torch) ** 2).item()
                logging.warning(f"Iteration {i+1}: MSE = {mse:.6f}")
            else:
                logging.info(f"Iteration {i+1}: hpim.mm and torch.mm are the same.")
        
        logging.info("test_mm_loop completed successfully")

if __name__ == "__main__":
    unittest.main()
