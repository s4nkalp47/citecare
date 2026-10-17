import pytest

from src.core.projects import get_project, list_projects


def test_medrag_is_registered():
    assert "medrag" in list_projects()


def test_medrag_config_has_required_fields():
    project = get_project("medrag")
    assert project.collection_name
    assert "{context_str}" in project.qa_prompt
    assert "{query_str}" in project.qa_prompt
    assert project.require_citations


def test_unknown_project_raises():
    with pytest.raises(KeyError):
        get_project("does-not-exist")
