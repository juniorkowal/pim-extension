import unittest
import logging
import hpim
import torch
import os
from tests.logging_func import setup_logging

class TestAdd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_logging(os.path.basename(__file__))

    def assertEqualWithTolerance(self, result, expected, tolerance=1e-2):
        """Helper method to assert equality with a tolerance."""
        self.assertTrue(
            torch.allclose(result, expected, rtol=tolerance, atol=tolerance),
            f"Result {result} does not match expected {expected} within tolerance {tolerance}"
        )

    def check_for_nans(self, tensor, test_case_name):
        """Helper method to check for NaNs in the result."""
        if torch.isnan(tensor).any():
            logging.warning(f"NaNs detected in the result for test case: {test_case_name}")

    def test_add_scalar_to_matrix(self):
        """Test case 1: Add random scalar to random 2x2 matrix."""
        logging.info("Running test_add_scalar_to_matrix")
        a = torch.randn(2, 2)
        b = torch.randn(1)
        result = hpim.ops.add(a, b)
        expected = torch.add(a, b)
        self.assertEqualWithTolerance(result, expected)
        self.check_for_nans(result, "Add random scalar to random 2x2 matrix")

    def test_add_matrix_to_matrix(self):
        """Test case 2: Add torch.randn 2x2 to another 2x2."""
        logging.info("Running test_add_matrix_to_matrix")
        a = torch.randn(2, 2)
        b = torch.randn(2, 2)
        result = hpim.ops.add(a, b)
        expected = torch.add(a, b)
        self.assertEqualWithTolerance(result, expected)
        self.check_for_nans(result, "Add torch.randn 2x2 to another 2x2")

    def test_add_large_matrices(self):
        """Test case 3: Add large matrices (1000x1000)."""
        logging.info("Running test_add_large_matrices")
        a = torch.randn(1000,1000)
        b = torch.randn(1000,1000)
        result = hpim.ops.add(a, b)
        expected = torch.add(a, b)
        self.assertEqualWithTolerance(result, expected)
        self.check_for_nans(result, "Add large matrices (1000x1000)")

    def test_additional_cases(self):
        """Additional test cases with varying shapes and broadcasting."""
        test_cases = [
            {"A_shape": (5, 4), "B_shape": (1,), "expected_shape": (5, 4)},
            {"A_shape": (5, 4), "B_shape": (4,), "expected_shape": (5, 4)},
            {"A_shape": (15, 3, 5), "B_shape": (15, 1, 5), "expected_shape": (15, 3, 5)},
            {"A_shape": (15, 3, 5), "B_shape": (3, 5), "expected_shape": (15, 3, 5)},
            {"A_shape": (15, 3, 5), "B_shape": (3, 1), "expected_shape": (15, 3, 5)},
            {"A_shape": (8, 1, 6, 1), "B_shape": (7, 1, 5), "expected_shape": (8, 7, 6, 5)},
            {"A_shape": (256, 256, 3), "B_shape": (3,), "expected_shape": (256, 256, 3)},
#            {"A_shape": (256, 256, 256), "B_shape": (256, 256, 256), "expected_shape": (256, 256, 256)},
        ]

        for idx, test_case in enumerate(test_cases, start=1):
            with self.subTest(test_case=test_case):
                logging.info(f"Running additional test case {idx}")
                a = torch.randn(test_case["A_shape"])
                b = torch.randn(test_case["B_shape"])
                result = hpim.ops.add(a, b)
                expected = torch.add(a, b)

                self.assertEqual(result.shape, test_case["expected_shape"],
                                 f"Shape mismatch in test case {idx}: {result.shape} != {test_case['expected_shape']}")

                self.assertEqualWithTolerance(result, expected)
                self.check_for_nans(result, f"Additional test case {idx}")

if __name__ == "__main__":
    unittest.main()
