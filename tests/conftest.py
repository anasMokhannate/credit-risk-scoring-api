import pytest

from credit_risk.config import Config
from credit_risk.spark import ensure_schemas, get_spark


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    session = get_spark(local_dir=tmp_path_factory.mktemp("spark"))
    yield session
    session.stop()


@pytest.fixture(scope="session")
def config(spark, tmp_path_factory):
    cfg = Config(raw_path=str(tmp_path_factory.mktemp("raw")))
    ensure_schemas(spark, cfg)
    return cfg
