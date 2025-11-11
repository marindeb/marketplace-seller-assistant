# config.py
import sys

import logging


APP_ENV = "prod"

LOG_FORMAT = "[%(asctime)s] %(levelname)s in %(name)s: %(message)s"


def setup_logging(level=logging.info):
    """Configure global logging."""
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
