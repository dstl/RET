"""Helper-Methods for space-based tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from parameterized import parameterized

if TYPE_CHECKING:
    from typing import Any

H_255 = 100
H_0 = -100
H_191 = (H_255 - H_0) * (191 / 255) + H_0
H_127 = (H_255 - H_0) * (127 / 255) + H_0

TEST_AGENTS_3D = [(-20, -20, -20), (-20, -20.05, -20.05), (65, 18, 32)]
REMOVAL_TEST_AGENTS_3D = [
    (-20, -20, -20),
    (-20, -20.05, -20.05),
    (65, 18, 32),
    (0, -11, 0),
    (20, 20, 20),
    (31, 41, 51),
    (55, 32, 42),
]
OUTSIDE_POSITIONS_3D = [(70, 10, 0), (30, 20, 20), (100, 10, -10)]


def get_test_name(cls: type, num: int, params_dict: dict[str, Any]) -> str:
    """Convert parametrised test case settings to string.

    Args:
        cls (type): Class under test
        num (int): Test number
        params_dict (dict[str, Any]): Test parameters

    Returns:
        str: Test name
    """
    return "%s_%s_%s_(%s, %s)" % (
        cls.__name__,
        num,
        parameterized.to_safe_name(str(params_dict["space_type"])),
        parameterized.to_safe_name(str(params_dict["pos"][0])),
        parameterized.to_safe_name(str(params_dict["pos"][1])),
    )
