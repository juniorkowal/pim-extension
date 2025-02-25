import os
import logging
from datetime import datetime

def setup_logging(test_name):
    """Set up logging to a file with a timestamped filename."""
    if logging.getLogger().hasHandlers():
        return

    script_dir = os.path.dirname(os.path.realpath(__file__))

    #test_folder = test_name.split('_')[1].split('.')[0]
    logs_dir = os.path.join(script_dir, 'logs'#, test_folder
    )
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(logs_dir, f"test_log_{timestamp}.log")

    plugin = os.getenv("PLUGIN", "1")

    if plugin == "1": # log only to console
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()]
        )
    else: # log to console and file
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),  # Log to file
                logging.StreamHandler()         # Log to console
            ]
        )

    logging.info(f"Running tests from: {test_name} ...")


def cleanup_logging():
    """Clean up logging by closing and removing all handlers."""
    for handler in logging.root.handlers[:]:
        # Close the handler if it has a `close` method
        if hasattr(handler, 'close'):
            handler.close()
        # Remove the handler from the root logger
        logging.root.removeHandler(handler)
    logging.info("Logging cleanup complete.")
