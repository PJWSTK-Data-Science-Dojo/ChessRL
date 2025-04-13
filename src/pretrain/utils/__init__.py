import logging
import sys
import os


def setup_logger(name='Chess', level=logging.INFO, log_file=None):
    """
    Sets up a logger with console and optional file output.
    """
    log = logging.getLogger(name)
    log.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s : %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
    return log


logger = setup_logger(name='PipelineLogger')
