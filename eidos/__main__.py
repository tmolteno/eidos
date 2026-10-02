"""Allow ``python -m eidos`` to run the command line interface."""

from .create_beam import cli

if __name__ == "__main__":
    cli()
