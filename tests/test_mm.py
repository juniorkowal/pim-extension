import logging
import torch
import hpim
import random
import os
from datetime import datetime

def setup_logging():
    script_dir = os.path.dirname(os.path.realpath(__file__))

    logs_dir = os.path.join(script_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    log_filename = datetime.now().strftime("mm-%m-%d-%H-%M.log")
    log_file_path = os.path.join(logs_dir, log_filename)

    logging.basicConfig(filename=log_file_path, 
                        level=logging.INFO, 
                        format='%(asctime)s - %(message)s')

    logging.info("Logging setup completed successfully.")

def test_mm_simple():
    logging.info("################################################")
    logging.info(f"START test_mm_simple.")
    
    mat = torch.randn(2, 2)
    
    result_hpim = hpim.ops.mm(mat, mat)
    result_torch = torch.mm(mat, mat)
    
    if torch.isnan(result_hpim).any():
        logging.warning(f"NaN detected in result of test_mm_simple (hpim).")
    
    logging.info(f"test_mm_simple hpim result: {result_hpim}")

    if torch.allclose(result_hpim, result_torch, atol=1e-6):
        logging.info(f"test_mm_simple success: hpim.mm and torch.mm results are the same.")
    else:
        logging.error(f"test_mm_simple mismatch: hpim.mm and torch.mm results are different.")
    
    logging.info(f"test_mm_simple torch result: {result_torch}")
    logging.info(f"END test_mm_simple")
    logging.info("################################################")

def test_mm_large(n: int = 300, m: int = 129, o: int = 317):
    logging.info("################################################")
    logging.info(f"START test_mm_large with n/m/o {n, m, o}.")
    
    mat1 = torch.randn(n, m)
    mat2 = torch.randn(m, o)
    
    result_hpim = hpim.ops.mm(mat1, mat2)
    result_torch = torch.mm(mat1, mat2)
    
    if torch.isnan(result_hpim).any():
        logging.warning(f"NaN detected in result of test_mm_large (hpim).")
    
    logging.info(f"test_mm_large hpim result, first number: {result_hpim[0, 0].item()}")
    
    if torch.allclose(result_hpim, result_torch, atol=1e-6):
        logging.info(f"test_mm_large success: hpim.mm and torch.mm results are the same.")
    else:
        logging.error(f"test_mm_large mismatch: hpim.mm and torch.mm results are different.")
    
    logging.info(f"test_mm_large torch result, first number: {result_torch[0, 0].item()}")
    logging.info(f"END test_mm_large")
    logging.info("################################################")

def test_mm_loop(num_iter: int = 10, mat_range: int = 300):
    logging.info("################################################")
    logging.info(f"START test_mm_loop with {num_iter} iterations and matrix range {mat_range}.")
    
    for i in range(num_iter):
        rows_mat1 = random.randint(1, mat_range)
        cols_mat1 = random.randint(1, mat_range)
        rows_mat2 = cols_mat1
        cols_mat2 = random.randint(1, mat_range)
        
        logging.info(f"\nIteration {i+1}: mat1 size: ({rows_mat1}, {cols_mat1}), mat2 size: ({rows_mat2}, {cols_mat2})")

        mat1 = torch.randn(rows_mat1, cols_mat1)
        mat2 = torch.randn(rows_mat2, cols_mat2)

        result_torch = torch.mm(mat1, mat2)
        result_hpim = hpim.ops.mm(mat1, mat2)

        logging.info(f"result_hpim first number: {result_hpim[0, 0].item()}, result_torch first_number: {result_torch[0, 0].item()}")

        if torch.isnan(result_hpim).any():
            logging.warning(f"NaN detected in result at iteration {i+1}")
        else:
            logging.info(f"Iteration {i+1}: First number of result_hpim is {result_hpim[0, 0].item()}")

        if not torch.allclose(result_hpim, result_torch, atol=1e-6):
            logging.error(f"Results mismatch at iteration {i+1}: hpim.mm and torch.mm are different.")
        else:
            logging.info(f"Iteration {i+1}: hpim.mm and torch.mm are the same.")
    logging.info("END test_mm_loop")
    logging.info("################################################")


if __name__ == "__main__":
    setup_logging()
    with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=True) as profiler:
        test_mm_simple()
        test_mm_large()
        test_mm_loop(num_iter=30)
    print(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
