# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
from subprocess import Popen

import pytest


@pytest.mark.parametrize(
    "binary",
    [
        "mipas_noxy",
        "python -m mipas_noxy",
    ],
)
def test_cli_help(binary):
    p = Popen(
        binary.split(" ") + [
            "-h",
        ],
    )
    p.communicate()
    p.wait()
    assert p.returncode == 0


@pytest.mark.parametrize(
    "binary",
    [
        "mipas_noxy",
        "python -m mipas_noxy",
    ],
)
def test_cli_dry_run(binary):
    p = Popen(
        binary.split(" ") + [
            "--config-file",
            "tests/test_NOx_config.toml",
            "-q",
            "-n",
        ],
    )
    p.communicate()
    p.wait()
    assert p.returncode == 0
