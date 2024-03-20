"""Test cases for Ret Register manager."""

import warnings

from ret.register_manager import RetRegisterManager


class MockObject:
    """A mock object for testing."""

    pass


class MockObject1(MockObject):
    """A mock object for testing."""

    pass


class MockObject2(MockObject):
    """A mock object for testing."""

    pass


class MockObject3(MockObject):
    """A mock object for testing."""

    def __init__(self, value: int) -> None:
        """Create a mock object with a value.

        Args:
            value (int): A value to be held by the mock object.
        """
        self.value = value


def test_mock_object_register_creation():
    """Test the creation of a RET Register."""
    mock_object_register = RetRegisterManager[MockObject]()
    assert mock_object_register.get_register_items() == []
    with warnings.catch_warnings(record=True) as w:
        assert mock_object_register.get_register_item("test") is None
    assert issubclass(w[-1].category, UserWarning)
    assert "'test' is unregistered. Returning None." == str(w[-1].message)


def test_registering_item():
    """Test the registering of an item into a RET Register."""
    mock_object_register = RetRegisterManager[MockObject]()
    mock_object_register.register("test", lambda: MockObject1())
    assert mock_object_register.get_register_items() == ["test"]
    assert isinstance(mock_object_register.get_register_item("test"), MockObject1)


def test_registering_multiple_items():
    """Test the registering of multiple items into a RET Register."""
    mock_object_register = RetRegisterManager[MockObject]()
    mock_object_register.register("test1", lambda: MockObject1())
    mock_object_register.register("test2", lambda: MockObject2())
    assert mock_object_register.get_register_items() == ["test1", "test2"]
    assert isinstance(mock_object_register.get_register_item("test1"), MockObject1)


def test_registering_item_same_name():
    """Test the overwriting of an item in a RET Register."""
    mock_object_register = RetRegisterManager[MockObject]()
    mock_object_register.register("test", lambda: MockObject1())
    with warnings.catch_warnings(record=True) as w:
        mock_object_register.register("test", lambda: MockObject2())
    assert issubclass(w[-1].category, UserWarning)
    assert "Overwriting the registered member: 'test'" == str(w[-1].message)

    assert mock_object_register.get_register_items() == ["test"]
    assert isinstance(mock_object_register.get_register_item("test"), MockObject2)


def test_invalid_register_request():
    """Test the requesting of an invalid item from a RET Register."""
    mock_object_register = RetRegisterManager[MockObject]()

    class InvalidMockObject:
        def __init__(self, test_arg) -> None:
            pass

    mock_object_register.register("invalid", lambda: InvalidMockObject())
    with warnings.catch_warnings(record=True) as w:
        mock_object_register.get_register_item("invalid")
    assert issubclass(w[-1].category, UserWarning)
    assert "Error in registered member: 'invalid'. Returning None." == str(w[-1].message)


def test_register_request_unregistered():
    """Test the requesting of an unregistered item from a RET Register."""
    mock_object_register = RetRegisterManager[MockObject]()
    with warnings.catch_warnings(record=True) as w:
        mock_object_register.get_register_item("absent")
    assert issubclass(w[-1].category, UserWarning)
    assert "'absent' is unregistered. Returning None." == str(w[-1].message)


def test_varied_config_register_items():
    """Test the registering of items configured differently in a RET Register."""
    mock_object_register = RetRegisterManager[MockObject]()
    mock_object_register.register("val_1", lambda: MockObject3(1))
    mock_object_register.register("val_2", lambda: MockObject3(2))
    assert mock_object_register.get_register_item("val_1").value == 1  # type: ignore
    assert mock_object_register.get_register_item("val_2").value == 2  # type: ignore


def test_none_register():
    """Test that the RetRegisterManager disallows adding 'None' items."""
    mock_object_register = RetRegisterManager[MockObject]()

    with warnings.catch_warnings(record=True) as w:
        mock_object_register.register("None", lambda: MockObject())

    assert len(w) == 1
    assert str(w[0].message) == "Cannot register an item to 'None'"


def test_none_access():
    """Test RetRegisterManager behaviour for requesting 'None'."""
    mock_object_register = RetRegisterManager[MockObject]()

    assert mock_object_register.get_register_item("None") is None
