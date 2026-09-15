import pytest
from mkt.databases import config


class TestConfig:
    def test_get_output_dir_exits_when_unset(self):
        """get_output_dir calls sys.exit(1) when OUTPUT_DIR is not set."""
        # clear any previously set value
        import os

        os.environ.pop("OUTPUT_DIR", None)
        with pytest.raises(SystemExit) as exc_info:
            config.get_output_dir()
        assert exc_info.value.code == 1

    def test_set_and_get_output_dir(self):
        """set_output_dir / get_output_dir round-trip."""
        config.set_output_dir("test")
        assert config.get_output_dir() == "test"

    def test_request_cache_default_none(self):
        """Request cache is None when not set."""
        import os

        os.environ.pop("REQUESTS_CACHE", None)
        assert config.maybe_get_request_cache() is None

    def test_set_and_get_request_cache(self):
        """set_request_cache / maybe_get_request_cache round-trip."""
        config.set_request_cache("test")
        assert config.maybe_get_request_cache() == "test"

    def test_get_cbioportal_instance_exits_when_unset(self):
        """get_cbioportal_instance calls sys.exit(1) when not set."""
        import os

        os.environ.pop("CBIOPORTAL_INSTANCE", None)
        with pytest.raises(SystemExit) as exc_info:
            config.get_cbioportal_instance()
        assert exc_info.value.code == 1

    def test_set_and_get_cbioportal_instance(self):
        """set_cbioportal_instance / get_cbioportal_instance round-trip."""
        config.set_cbioportal_instance("test")
        assert config.get_cbioportal_instance() == "test"

    def test_cbioportal_token_default_none(self):
        """cBioPortal token is None when not set."""
        import os

        os.environ.pop("CBIOPORTAL_TOKEN", None)
        assert config.maybe_get_cbioportal_token() is None

    def test_set_and_get_cbioportal_token(self):
        """set_cbioportal_token / maybe_get_cbioportal_token round-trip."""
        config.set_cbioportal_token("test")
        assert config.maybe_get_cbioportal_token() == "test"
