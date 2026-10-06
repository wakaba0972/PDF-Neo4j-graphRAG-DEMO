from manual_graphrag.env_store import load_env, save_env


def test_env_round_trip_preserves_special_characters(tmp_path) -> None:
    path = tmp_path / ".env"
    save_env(
        {
            "NEO4J_PASSWORD": "p#a ss=word",
            "MODEL_OPENAI_API_KEY": "sk-test",
        "MODEL_OPENAI_API_BASE": "https://api.openai.com/v1",
        },
        path,
    )
    loaded = load_env(path)
    assert loaded["NEO4J_PASSWORD"] == "p#a ss=word"
    assert loaded["MODEL_OPENAI_API_KEY"] == "sk-test"
    assert loaded["MODEL_OPENAI_API_BASE"] == "https://api.openai.com/v1"


def test_save_env_preserves_unmanaged_values(tmp_path) -> None:
    path = tmp_path / ".env"
    path.write_text("CUSTOM_VALUE=keep\nNEO4J_URI=old\n", encoding="utf-8")
    save_env({"NEO4J_URI": "bolt://new"}, path)
    content = path.read_text(encoding="utf-8")
    assert "CUSTOM_VALUE=keep" in content
    assert load_env(path)["NEO4J_URI"] == "bolt://new"


def test_concurrent_connection_updates_preserve_openai_credentials(tmp_path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    path = tmp_path / ".env"
    updates = {
        "MODEL_OPENAI_API_KEY": "llm-openai-key",
        "EMBEDDING_OPENAI_API_KEY": "embedding-openai-key",
    }
    barrier = Barrier(len(updates))
    def write(item):
        key, value = item
        barrier.wait()
        save_env({key: value}, path)
    with ThreadPoolExecutor(max_workers=len(updates)) as executor:
        list(executor.map(write, updates.items()))
    loaded = load_env(path)
    assert all(loaded[key] == value for key, value in updates.items())


def test_save_env_removes_ollama_and_voyage_keys(tmp_path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "MODEL_OLLAMA_API_KEY=legacy\nEMBEDDING_VOYAGE_API_KEY=legacy\n",
        encoding="utf-8",
    )
    save_env({"MODEL_OPENAI_API_KEY": "new"}, path)
    content = path.read_text(encoding="utf-8")
    assert "OLLAMA" not in content
    assert "VOYAGE" not in content


def test_save_env_removes_legacy_model_and_profile_keys(tmp_path) -> None:
    path = tmp_path / ".env"
    path.write_text('BUILD_MODEL="old"\nMODEL_SERVICE_PROFILES="{}"\n', encoding="utf-8")
    save_env({"MODEL_OPENAI_API_KEY": "new"}, path)
    content = path.read_text(encoding="utf-8")
    assert "BUILD_MODEL" not in content
    assert "MODEL_SERVICE_PROFILES" not in content
    assert 'MODEL_OPENAI_API_KEY="new"' in content
