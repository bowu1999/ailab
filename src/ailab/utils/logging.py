import logging, os


def get_logger(name='openmmlab', log_file=None):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler(); ch.setFormatter(fmt); logger.addHandler(ch)
        if log_file:
            fh = logging.FileHandler(log_file); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger