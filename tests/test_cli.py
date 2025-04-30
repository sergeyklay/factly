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
