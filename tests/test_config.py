import pytest
class NotInRangeError(Exception):
    def __init__(self, message="Error message"):
        self.message = message
        super().__init__(self.message)

def test_generaic():
    a = 3
    b = 3
    assert a == b

def test_withexception():
    a = 15  # Change the value of 'a' to be within the range (10, 20)
    with pytest.raises(NotInRangeError):
        if a in range(10, 20):
            raise NotInRangeError("Value is not in range")
