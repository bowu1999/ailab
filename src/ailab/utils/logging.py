import os
import logging
from tqdm import tqdm

class TqdmToLogger(logging.Handler):
    """
    This class is a handler for the logging module that redirects log messages to tqdm.write().
    This allows for proper formatting of progress bars when logging.
    """
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger(name='AiLab', log_file=None, stream_log_level=logging.INFO, file_log_level=logging.INFO):
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers to the same logger
    if not logger.handlers:
        # Set the minimum level of logs that this logger will handle
        logger.setLevel(min(stream_log_level, file_log_level))

        # Define the log format
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Add the custom handler for tqdm
        tqdm_handler = TqdmToLogger()
        tqdm_handler.setLevel(stream_log_level)
        tqdm_handler.setFormatter(fmt)
        logger.addHandler(tqdm_handler)

        # If a log file is specified, create file handler and set level and formatter
        if log_file:
            # Ensure the directory for the log file exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setLevel(file_log_level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    
    return logger