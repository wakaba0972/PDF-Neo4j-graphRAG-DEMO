from app import launch_settings


def test_launch_settings_bind_all_interfaces_at_port_8080(monkeypatch) -> None:
    monkeypatch.setenv("GRADIO_SERVER_NAME", "127.0.0.1")
    monkeypatch.setenv("GRADIO_SERVER_PORT", "8000")
    monkeypatch.delenv("GRADIO_INBROWSER", raising=False)

    assert launch_settings() == {
        "server_name": "0.0.0.0",
        "server_port": 8080,
        "share": False,
        "inbrowser": True,
        "_frontend": False,
    }


def test_launch_settings_ignore_server_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("GRADIO_SERVER_NAME", "127.0.0.1")
    monkeypatch.setenv("GRADIO_SERVER_PORT", "9000")
    monkeypatch.setenv("GRADIO_INBROWSER", "false")

    settings = launch_settings()

    assert settings["server_name"] == "0.0.0.0"
    assert settings["server_port"] == 8080
    assert settings["inbrowser"] is False
