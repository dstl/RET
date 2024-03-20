"""Test the find and copy map method with different folder setups."""

import os
import pathlib
from tempfile import TemporaryDirectory
from ret.batchrunner import FixedReportingBatchRunner
from unittest import TestCase
from ret.utilities.save_utilities import add_datetime_stamp


class BatchrunnerFindCopyTest(TestCase):
    """Test that map is copied correctly."""

    def test_map_name_given(self):
        """Test _find_map_file when a map name is given."""
        with TemporaryDirectory() as td:
            map_path = pathlib.Path(td, "base_map.png")
            map_path.write_text("")
            br = FixedReportingBatchRunner(
                model_cls=None,
                output_path=td,
            )
            assert (
                pathlib.Path(br._find_map_file(map_name="base_map.png", image_path=td)) == map_path
            )

    def test_no_map(self):
        """Test correct warning shows when no map file found."""
        with TemporaryDirectory() as td:
            br = FixedReportingBatchRunner(
                model_cls=None,
                output_path=td,
            )

            with self.assertWarns(Warning) as w:
                br.find_and_copy_map()

            assert "No map file found to copy in: " in str(w.warnings[0].message)

    def test_incorrect_map_name(self):
        """Test map file gets copied if it is not in folder."""
        with TemporaryDirectory() as td:
            br = FixedReportingBatchRunner(
                model_cls=None,
                output_path=td,
            )
            with self.assertWarns(Warning) as w:
                br.find_and_copy_map(map_name="fake_map")

            assert "No map file found by the name: " in str(w.warnings[0].message)

    def test_with_map_no_folders(self):
        """Test map file gets copied if it is not in folder."""
        with TemporaryDirectory() as td:
            map_path = pathlib.Path(os.path.dirname(os.path.abspath(__file__)), "base_map.png")
            map_path.write_text("")
            br = FixedReportingBatchRunner(
                model_cls=None,
                output_path=td,
            )
            outpath = pathlib.Path(add_datetime_stamp(td), "assets")
            outpath.mkdir(parents=True, exist_ok=True)
            br.find_and_copy_map()

            outfile = os.path.join(outpath, "Base_map.png")

            self.assertTrue(os.path.isfile(outfile))

            os.remove(map_path)

    def test_with_map_2_folders(self):
        """Test map file gets copied if there are 2 possible folder it could be in."""
        with TemporaryDirectory() as td:
            p = pathlib.Path(os.path.dirname(os.path.abspath(__file__)), "test_save", "images")
            p.mkdir(parents=True, exist_ok=True)
            q = pathlib.Path(os.path.dirname(os.path.abspath(__file__)), "wrong_folder")
            q.mkdir(parents=True, exist_ok=True)
            map_path = pathlib.Path(p, "Base_map.png")
            map_path.write_text("")
            br = FixedReportingBatchRunner(
                model_cls=None,
                output_path=td,
            )
            outpath = pathlib.Path(add_datetime_stamp(td), "assets")
            outpath.mkdir(parents=True, exist_ok=True)
            br.find_and_copy_map()

            outfile = os.path.join(outpath, "Base_map.png")

            self.assertTrue(os.path.isfile(outfile))

            os.remove(map_path)
            os.rmdir(p)
            os.rmdir(q)
            os.rmdir(os.path.dirname(p))

    def test_with_map_1_folders(self):
        """Test map file gets copied if there are 1 possible folder it could be in."""
        with TemporaryDirectory() as td:
            p = pathlib.Path(os.path.dirname(os.path.abspath(__file__)), "correct_folder", "images")
            p.mkdir(parents=True, exist_ok=True)
            map_path = pathlib.Path(p, "base_map.png")
            map_path.write_text("")
            br = FixedReportingBatchRunner(
                model_cls=None,
                output_path=td,
            )
            outpath = pathlib.Path(add_datetime_stamp(td), "assets")
            outpath.mkdir(parents=True, exist_ok=True)
            br.find_and_copy_map()

            outfile = os.path.join(outpath, "Base_map.png")

            self.assertTrue(os.path.isfile(outfile))

            os.remove(map_path)
            os.rmdir(p)
            os.rmdir(os.path.dirname(p))
