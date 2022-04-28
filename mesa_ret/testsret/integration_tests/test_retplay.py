"""RetPlay integration tests."""
import pathlib
from tempfile import TemporaryDirectory

from dash import Dash
from retplay import create_app


def test_retplay_creation(capsys):
    """Smoke test to check that the app can be created without throwing errors.

    Args:
        capsys: Print output capture
    """
    with TemporaryDirectory() as td:
        p = pathlib.Path(td, "assets/icons")
        p.mkdir(parents=True, exist_ok=True)
        playback_path = pathlib.Path(td, "playback.json")
        with open("./mesa_ret/testsret/testing_assets/playback.json", "r") as f:
            json_text = f.read()
        playback_path.write_text(json_text)
        map_path = pathlib.Path(td, "assets/base_map.png")
        map_path.write_text("")
        icon_path = pathlib.Path(p, "test_icon.svg")
        icon_path.write_text("")

        app = create_app(td)

        out, _ = capsys.readouterr()

        assert isinstance(app, Dash)
        assert "Loading model from playback file in folder" in str(out)
