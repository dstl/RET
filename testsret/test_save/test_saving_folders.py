"""Test the get_latest_subfolder method."""
from tempfile import TemporaryDirectory
import pathlib
from time import sleep
from datetime import datetime
import os
from ret.visualisation.json_icon_handler import IconCopier
import warnings

from ret.utilities.save_utilities import (
    get_latest_subfolder,
    add_datetime_stamp,
    validate_save_path,
)


def test_add_datetime_stamp_no_stamp():
    """Test to see if add_datetime_stamp will add stamp to path with no stamp."""
    path = r"C:\Users\Tester\Downloads"
    assert len(add_datetime_stamp(path)) > len(path)


def test_add_datetime_stamp_yes_stamp():
    """Test to see if add_datetime_stamp will not add stamp to path with stamp."""
    path = r"C:\Users\Tester\Downloads"
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(path, current_datetime)
    assert add_datetime_stamp(path) == path


def test_validate_existing_folder():
    """Test to see if validate save path will return path if path exists."""
    with TemporaryDirectory() as td:
        assert td == validate_save_path(td)


def test_validate_existing_subfolder():
    """Test to see if validate save path will return path if path exists."""
    with TemporaryDirectory() as td:
        correct_subfolder = os.path.join(td, "correct")
        incorrect_subfolder = os.path.join(td, "incorrect")
        os.makedirs(correct_subfolder, exist_ok=True)
        assert correct_subfolder == validate_save_path(incorrect_subfolder)


def test_2_sub_folders():
    """Test for the get latest subfolder where 2 folders exits ."""
    with TemporaryDirectory() as td:
        p = pathlib.Path(td, "old")
        p.mkdir(parents=True, exist_ok=True)
        sleep(0.2)
        q = pathlib.Path(td, "new")
        q.mkdir(parents=True, exist_ok=True)

        assert get_latest_subfolder(td) == "new"


def test_no_sub_folders():
    """Test for the get latest subfolder where no folders exits ."""
    with TemporaryDirectory() as td:
        assert get_latest_subfolder(td) is None


def test_no_sub_folders_icon_copier():
    """Test icon copier for path that doesn't exist."""
    with TemporaryDirectory() as td:
        copier = IconCopier(output_folder_name=td + "/fakefolder")

        with warnings.catch_warnings(record=True) as w:
            copier.copy_icon(icon_path=td + "/fakefolder")
        assert "icon path either doesn't exist or is not a file." in str(w[0].message)
