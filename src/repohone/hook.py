"""Hook entry point inside the package, so an installed Core can be invoked as
`python -m repohone.hook` without naming the directory it was installed from."""
import sys

from .cli import main


def run() -> int:
    return main(sys.argv[1:] or ["hook"])


if __name__ == "__main__":
    sys.exit(run())
