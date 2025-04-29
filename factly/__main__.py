import sys

from factly.cli import main


def init() -> None:
    """
    Run factly.cli.main() when current file is executed by an interpreter.

    This function ensures that the CLI main function is only executed when this
    file is run directly, not when imported as a module.

    The :func:`sys.exit` function is called with the return value of
    :func:`factly.cli.main`, following standard UNIX program conventions
    for exit codes.
    """
    if __name__ == "__main__":
        sys.exit(main())


init()
