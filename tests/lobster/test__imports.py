from lightning_utilities.core.imports import RequirementCache
from lobster import _PRESCIENT_AVAILABLE


class TestImports:
    def test_prescient_available(self):
        print(_PRESCIENT_AVAILABLE)

        assert _PRESCIENT_AVAILABLE

    def test_dummy_library_available(self):
        dummy_available = RequirementCache("dummy-library")
        print(dummy_available)

        assert not dummy_available
