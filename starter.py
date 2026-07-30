import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.cli import cli

cli()
