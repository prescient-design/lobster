from lightning_utilities.core.imports import RequirementCache


class TestImports:
    def test_dummy_library_available(self):
        dummy_available = RequirementCache("dummy-library")
        print(dummy_available)

        assert not dummy_available
