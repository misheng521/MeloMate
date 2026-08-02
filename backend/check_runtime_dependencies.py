"""Validate dependency versions that cannot be expressed as import checks."""

from importlib.metadata import PackageNotFoundError, version
import re


MINIMUM_MCP_VERSION = (1, 28)


def validate_mcp_version() -> tuple[bool, str]:
    try:
        installed = version("mcp")
    except PackageNotFoundError:
        return False, "MCP Python SDK is not installed."

    match = re.match(r"^(\d+)\.(\d+)", installed)
    if match is None:
        return False, f"MCP Python SDK has an unsupported version string: {installed}"

    major_minor = (int(match.group(1)), int(match.group(2)))
    if major_minor[0] != 1 or major_minor < MINIMUM_MCP_VERSION:
        return (
            False,
            f"MCP Python SDK {installed} is incompatible; MeloMate requires >=1.28,<2.",
        )
    return True, f"MCP Python SDK {installed} verified."


def main() -> int:
    valid, message = validate_mcp_version()
    print(message)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
