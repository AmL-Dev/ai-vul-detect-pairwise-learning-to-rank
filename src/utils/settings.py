""" Common configuration for the project.
"""

import logging.config

from pathlib import Path
# `path.parents[1]` is the same as `path.parent.parent`
ROOT_DIR = Path(__file__).resolve().parents[2]

""" str: Project root directory.
"""

# Logging configuration
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
LOGGER = logging.getLogger(__name__)
""" logging.Logger: `Logger` instance used throughout the application.
"""
