"""Test retplay methods."""
import inspect
import os
import pathlib
import shutil
from tempfile import TemporaryDirectory
from unittest import TestCase
from warnings import catch_warnings

from retplay.app import copy_playback_assets, validate_playback_data


class RetPlayPlaybackValidationTest(TestCase):
    """Test the input validation method for RetPlay."""

    def test_no_playback_file(self):
        """Test passing in a folder that does not contain a playback file."""
        with TemporaryDirectory() as td:
            p = pathlib.Path(td, "assets", "icons")
            p.mkdir(parents=True, exist_ok=True)
            map_path = pathlib.Path(td, "assets", "base_map.png")
            map_path.write_text("")
            icon_path = pathlib.Path(p, "test_icon.svg")
            icon_path.write_text("")

            with self.assertRaises(TypeError) as e:
                validate_playback_data(td)
            self.assertEqual(
                "Playback file does not exist in selected folder.",
                str(e.exception),
            )

    def test_no_assets_dir(self):
        """Test passing in a folder that does not contain an assets subdirectory."""
        with TemporaryDirectory() as td:
            playback_path = pathlib.Path(td, "playback.json")
            playback_path.write_text("")

            with self.assertRaises(TypeError) as e:
                validate_playback_data(td)
            self.assertEqual(
                "Assets folder does not exist in selected folder.",
                str(e.exception),
            )

    def test_no_icons_dir(self):
        """Test passing in a folder that does not contain an icons subdirectory."""
        with TemporaryDirectory() as td:
            p = pathlib.Path(td, "assets")
            p.mkdir(parents=True, exist_ok=True)
            playback_path = pathlib.Path(td, "playback.json")
            playback_path.write_text("")
            map_path = pathlib.Path(p, "base_map.png")
            map_path.write_text("")

            with catch_warnings(record=True) as w:
                validate_playback_data(td)
                assert len(w) == 1
                assert "Icons folder does not exist in selected folder." in str(w[0].message)

    def test_no_map_file(self):
        """Test passing in a folder that does not contain a map image file."""
        with TemporaryDirectory() as td:
            p = pathlib.Path(td, "assets", "icons")
            p.mkdir(parents=True, exist_ok=True)
            playback_path = pathlib.Path(td, "playback.json")
            playback_path.write_text("")
            icon_path = pathlib.Path(p, "test_icon.svg")
            icon_path.write_text("")

            with catch_warnings(record=True) as w:
                validate_playback_data(td)
                assert len(w) == 1
                assert "No map file present. No image will be used." in str(w[0].message)

    def test_no_icon_files(self):
        """Test passing in a folder that does not contain any icon image files."""
        with TemporaryDirectory() as td:
            p = pathlib.Path(td, "assets", "icons")
            p.mkdir(parents=True, exist_ok=True)
            playback_path = pathlib.Path(td, "playback.json")
            playback_path.write_text("")
            map_path = pathlib.Path(td, "assets", "base_map.png")
            map_path.write_text("")

            with catch_warnings(record=True) as w:
                validate_playback_data(td)
                assert len(w) == 1
                assert "No icon images present. Default icons will be used." in str(w[0].message)

    def test_not_folder(self):
        """Test passing in a path to a folder that does not exist."""
        with self.assertRaises(TypeError) as e:
            validate_playback_data("C:/not_a_folder")
        self.assertEqual(
            "The folder selected does not exist.",
            str(e.exception),
        )


class RetPlayCopyAssetsTest(TestCase):
    """Test that playback assets are copied correctly."""

    def test_copy_method(self):
        """Test that a full assets tree is copied correctly."""
        with TemporaryDirectory() as td:
            p = pathlib.Path(td, "assets", "icons")
            p.mkdir(parents=True, exist_ok=True)
            playback_path = pathlib.Path(td, "playback.json")
            playback_path.write_text("")
            map_path = pathlib.Path(td, "assets", "base_map.png")
            map_path.write_text("")
            icon_path = pathlib.Path(p, "test_icon.svg")
            icon_path.write_text("")

            folder = copy_playback_assets(td)
            copied_path = pathlib.Path(
                os.path.dirname(inspect.getfile(copy_playback_assets)), folder
            )

        copied_playback = str(pathlib.Path(copied_path, "playback.json"))
        copied_map = str(pathlib.Path(copied_path, "assets", "base_map.png"))
        copied_icon = str(pathlib.Path(copied_path, "assets", "icons", "test_icon.svg"))

        copied_file_list = []
        for path, _, files in os.walk(copied_path):
            for name in files:
                copied_file_list.append(os.path.join(path, name))

        assert copied_playback in copied_file_list
        assert copied_map in copied_file_list
        assert copied_icon in copied_file_list

        shutil.rmtree(copied_path)  # removes folder used only for testing
