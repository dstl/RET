"""Test cases for Ret Playback Writer Register."""

import warnings

import ret.visualisation
from ret.testing.mocks import MockModel3d
from ret.visualisation import get_playback_writer_register
from ret.visualisation.playback_writer import PlaybackWriter


class MockPlaybackWriter(PlaybackWriter):
    """A mock playback writer for testing."""

    mock_model = MockModel3d()

    def model_start(self, model=None) -> None:
        """A mock method to do nothing for testing.

        Args:
            model ([type], optional): Present to satisfy signature. Defaults to None.

        Returns:
            None: Returns None.
        """
        return super().model_start(self.mock_model)

    def model_step(self, model=None) -> None:
        """A mock method to do nothing for testing.

        Args:
            model ([type], optional): Present to satisfy signature. Defaults to None.

        Returns:
            None: Returns None.
        """
        return super().model_step(self.mock_model)

    def model_finish(self) -> None:
        """A mock method to do nothing for testing.

        Returns:
            None: Returns None.
        """
        return super().model_finish()


class MockPlaybackWriter1(MockPlaybackWriter):
    """A mock playback writer for testing."""

    pass


class MockPlaybackWriter2(MockPlaybackWriter):
    """A mock playback writer for testing."""

    pass


class MockPlaybackWriter3(MockPlaybackWriter):
    """A mock playback writer with argument for testing."""

    def __init__(self, value: int) -> None:
        """Create a mock playback writer with a value.

        Args:
            value (int): A value to be held by the mock playback writer.
        """
        self.value = value
        super().__init__()


def test_playback_writer_register_creation():
    """Test the creation of a playback writer."""
    ret.visualisation.__writer_register__ = None
    playback_writer_register = get_playback_writer_register()
    assert playback_writer_register.get_register_items() == ["JsonWriter"]
    with warnings.catch_warnings(record=True) as w:
        assert playback_writer_register.get_register_item("test") is None
    assert issubclass(w[-1].category, UserWarning)
    assert "'test' is unregistered. Returning None." == str(w[-1].message)


def test_playback_writer_registering_item():
    """Test the registering of a playback writer."""
    ret.visualisation.__writer_register__ = None
    playback_writer_register = get_playback_writer_register()
    playback_writer_register.register("test", lambda: MockPlaybackWriter1())
    assert playback_writer_register.get_register_items() == ["JsonWriter", "test"]
    assert isinstance(playback_writer_register.get_register_item("test"), MockPlaybackWriter1)


def test_playback_writer_registering_multiple_items():
    """Test the registering of multiple playback writers."""
    ret.visualisation.__writer_register__ = None
    playback_writer_register = get_playback_writer_register()
    playback_writer_register.register("test1", lambda: MockPlaybackWriter1())
    playback_writer_register.register("test2", lambda: MockPlaybackWriter2())
    assert playback_writer_register.get_register_items() == ["JsonWriter", "test1", "test2"]
    assert isinstance(playback_writer_register.get_register_item("test1"), MockPlaybackWriter1)


def test_playback_writer_registering_item_same_name():
    """Test the overwriting of a playback writer."""
    ret.visualisation.__writer_register__ = None
    playback_writer_register = get_playback_writer_register()
    playback_writer_register.register("test", lambda: MockPlaybackWriter1())
    with warnings.catch_warnings(record=True) as w:
        playback_writer_register.register("test", lambda: MockPlaybackWriter2())
    assert issubclass(w[-1].category, UserWarning)
    assert "Overwriting the registered member: 'test'" == str(w[-1].message)

    assert playback_writer_register.get_register_items() == ["JsonWriter", "test"]
    assert isinstance(playback_writer_register.get_register_item("test"), MockPlaybackWriter2)


def test_playback_writer_invalid_register_request():
    """Test the return of an invalid request to the playback register."""
    ret.visualisation.__writer_register__ = None
    playback_writer_register = get_playback_writer_register()

    class InvalidMockObject:
        def __init__(self, test_arg) -> None:
            pass

    playback_writer_register.register("invalid", lambda: InvalidMockObject())
    with warnings.catch_warnings(record=True) as w:
        playback_writer_register.get_register_item("invalid")
    assert issubclass(w[-1].category, UserWarning)
    assert "Error in registered member: 'invalid'. Returning None." == str(w[-1].message)


def test_playback_writer_register_request_unregistered():
    """Test the request of an unregistered playback writer."""
    global __writer_register__
    __writer_register__ = None
    playback_writer_register = get_playback_writer_register()
    with warnings.catch_warnings(record=True) as w:
        playback_writer_register.get_register_item("absent")
    assert issubclass(w[-1].category, UserWarning)
    assert "'absent' is unregistered. Returning None." == str(w[-1].message)


def test_playback_writer_varied_config_register_items():
    """Test the storing of differently configured playback writers."""
    global __writer_register__
    __writer_register__ = None
    playback_writer_register = get_playback_writer_register()
    playback_writer_register.register("val_1", lambda: MockPlaybackWriter3(1))
    playback_writer_register.register("val_2", lambda: MockPlaybackWriter3(2))
    assert playback_writer_register.get_register_item("val_1").value == 1  # type: ignore
    assert playback_writer_register.get_register_item("val_2").value == 2  # type: ignore
