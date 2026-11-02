from ai_proj.core.config import settings


def test_app_name():
    assert settings.app_name == "AI Project"