"""The generator modules import each other by bare name (`import cv_templates`),
because they are run as scripts from this folder. Tests need the same path."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
