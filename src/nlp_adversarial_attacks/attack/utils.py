import logging
import sys


def cmd_args_to_yaml(cmd_args, outfile_name, ignore_list=None):
    """
    Takes cmd_args, an argparse.Namespace object, and writes the values to a file
    in YAML format. Some parameter values might not need to be saved, so you can
    pass a list of parameter names as the ignore_list, and the values for these
    parameter names will not be saved to the YAML file.
    """
    if ignore_list is None:
        ignore_list = []
    cmd_args_dict = vars(cmd_args)
    with open(outfile_name, "w") as yaml_outfile:
        for parameter, value in cmd_args_dict.items():
            # don't write the parameter value if parameter in the
            # ignore list or the value of the parameter is None
            if parameter in ignore_list or value is None:
                continue
            else:
                # write boolean values as 1's and 0's
                if isinstance(value, bool):
                    value = int(value)
                yaml_outfile.write(f"{parameter}: {value}\n")


def get_logger(filename=""):
    """
    Return a logger object to easily save textual output.
    """

    logger = logging.getLogger()
    logger.handlers = []  # clear previous handlers
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(message)s")
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    if filename:
        log_handler = logging.FileHandler(filename, mode="w")
        log_handler.setLevel(logging.INFO)
        log_handler.setFormatter(formatter)
        logger.addHandler(log_handler)

    return logger


def remove_logger(logger):
    """
    Remove handlers from logger.
    """
    logger.handlers = []
