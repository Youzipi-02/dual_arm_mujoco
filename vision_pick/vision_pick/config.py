"""Read installed parameter defaults; launch overrides remain authoritative."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
import yaml


def load_defaults(package, filename):
    path = Path(get_package_share_directory(package)) / 'config' / filename
    return yaml.safe_load(path.read_text())['/**']['ros__parameters']

