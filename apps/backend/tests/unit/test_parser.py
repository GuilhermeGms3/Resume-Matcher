"""Unit tests for resume parsing helpers."""

from app.schemas import ResumeData
import pytest

from app.services.parser import (
    normalize_llm_resume_data,
    parse_resume_locally,
    validate_llm_resume_completeness,
)


def test_normalize_llm_resume_data_coerces_null_required_strings():
    data = normalize_llm_resume_data(
        {
            "personalInfo": {"name": "Jane Doe", "phone": None},
            "personalProjects": [{"name": "Tool", "role": "Owner", "years": None}],
        }
    )

    validated = ResumeData.model_validate(data)

    assert validated.personalInfo.phone == ""
    assert validated.personalProjects[0].years == ""


def test_parse_resume_locally_extracts_basic_resume_fields():
    data = parse_resume_locally(
        "Guilherme Aires Gomes\n"
        "guilherme@example.com\n"
        "Area: Infraestrutura\n"
        "- Docker e Proxmox\n"
    )

    assert data["personalInfo"]["name"] == "Guilherme Aires Gomes"
    assert data["personalInfo"]["email"] == "guilherme@example.com"
    assert data["personalInfo"]["title"] == "Infraestrutura"
    assert data["workExperience"][0]["description"] == ["Docker e Proxmox"]


def test_validate_llm_resume_completeness_rejects_empty_structured_bullets():
    parsed = {
        "personalInfo": {"name": "Guilherme"},
        "summary": "",
        "workExperience": [],
        "personalProjects": [],
        "customSections": {
            "skills": {
                "sectionType": "itemList",
                "items": [{"title": "Administrei Active Directory"}],
            }
        },
    }

    with pytest.raises(ValueError):
        validate_llm_resume_completeness(parsed, "Guilherme\n- Administrei Active Directory")
