import pytest

from rarecare.config import Config
from rarecare.kg.graph import default_graph
from rarecare.pipeline import RareCarePipeline


@pytest.fixture(scope="session")
def kg():
    return default_graph()


@pytest.fixture(scope="session")
def pipe(kg):
    return RareCarePipeline(Config(), kg)
