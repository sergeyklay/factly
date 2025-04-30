import subprocess
import sys

import pytest
from click.testing import CliRunner

from factly import __copyright__, __version__
from factly.cli import cli


def test_version_option():
    """Test that the --version option prints the correct version and exits."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert f"factly {__version__}" in result.output
    assert __copyright__ in result.output
    assert "This is free software" in result.output
    assert "warranty" in result.output


@pytest.mark.skipif(
    sys.platform == "win32", reason="Command execution differs on Windows"
)
def test_version_option_module():
    """Test that the --version option works when running as a module."""
    result = subprocess.run(
        [sys.executable, "-m", "factly", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert f"factly {__version__}" in result.stdout
    assert __copyright__ in result.stdout
    assert "This is free software" in result.stdout
    assert "warranty" in result.stdout
