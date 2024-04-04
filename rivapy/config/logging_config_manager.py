# -*- coding: utf-8 -*-


import logging
from logging import config as log_config, basicConfig as log_basicConfig
from os import getenv as os_getenv, path as os_path
from sys import stdout

from coloredlogs import install as col_logs_install
from yaml import safe_load as yaml_save_load


def setup_logging(
    config_path: str = "config/logging_config.yaml",  # TODO: correct this line
    log_level: int = logging.INFO,
    env_key: str = "LOG_CFG",
):
    """
    Setup logging based on logging configuration for specified logging level.

    Args:
        config_path (str, optional): Logging configuration path (file in yaml format).
                                     Defaults to 'config/logging_config.yaml'
        log_level (int, optional): Log level for which the logging is to be setup.
                                   Defaults to logging.INFO
        env_key (str, optional): Logging config path set in environment variable.
                                 Defaults to 'LOG_CFG'

    Returns:
        n/a
    """
    path = config_path
    value = os_getenv(env_key, None)
    if value:
        path = value
    if os_path.exists(path):
        with open(path, "rt") as file:
            try:
                yaml_config = yaml_save_load(file.read())
                log_config.dictConfig(yaml_config)
                col_logs_install()
            except Exception as exception:
                print(exception)
                print("Error in Logging Configuration. Using default configs")
                log_basicConfig(level=log_level, stream=stdout)
                col_logs_install(level=log_level)
    else:
        log_basicConfig(level=log_level, stream=stdout)
        col_logs_install(level=log_level)
        print("Failed to load configuration file. Using default configs")
