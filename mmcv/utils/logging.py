
import logging
import torch.distributed as dist
logger_initialized = {

}


def get_logger(name, log_file=None, log_level=logging.INFO):
    'Initialize and get a logger by name.\n\n    If the logger has not been initialized, this method will initialize the\n    logger by adding one or two handlers, otherwise the initialized logger will\n    be directly returned. During initialization, a StreamHandler will always be\n    added. If `log_file` is specified and the process rank is 0, a FileHandler\n    will also be added.\n\n    Args:\n        name (str): Logger name.\n        log_file (str | None): The log filename. If specified, a FileHandler\n            will be added to the logger.\n        log_level (int): The logger level. Note that only the process of\n            rank 0 is affected, and other processes will set the level to\n            "Error" thus be silent most of the time.\n\n    Returns:\n        logging.Logger: The expected logger.\n    '
    logger = logging.getLogger(name)
    if (name in logger_initialized):
        return logger
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]
    if (dist.is_available() and dist.is_initialized()):
        rank = dist.get_rank()
    else:
        rank = 0
    if ((rank == 0) and (log_file is not None)):
        file_handler = logging.FileHandler(log_file, 'w')
        handlers.append(file_handler)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    if (rank == 0):
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)
    logger_initialized[name] = True
    return logger


def print_log(msg, logger=None, level=logging.INFO):
    'Print a log message.\n\n    Args:\n        msg (str): The message to be logged.\n        logger (logging.Logger | str | None): The logger to be used.\n            Some special loggers are:\n            - "silent": no message will be printed.\n            - other str: the logger obtained with `get_root_logger(logger)`.\n            - None: The `print()` method will be used to print log messages.\n        level (int): Logging level. Only available when `logger` is a Logger\n            object or "root".\n    '
    if (logger is None):
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif (logger == 'silent'):
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(''.join(
            ['logger should be either a logging.Logger object, str, "silent" or None, but got ', '{}'.format(type(logger))]))
