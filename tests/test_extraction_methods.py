from pathlib import Path

from manual_graphrag.extraction_methods import (
    EXTRACTION_METHODS,
    load_extraction_method_specs,
)


def test_default_methods_are_loaded_from_yaml_and_implementations() -> None:
    assert set(EXTRACTION_METHODS.specs) == {"typed_name", "name_only"}
    assert EXTRACTION_METHODS.get("typed_name").requires_entity_type is True
    assert EXTRACTION_METHODS.get("name_only").requires_entity_type is False


def test_name_only_method_merges_name_variants_without_entity_type() -> None:
    method = EXTRACTION_METHODS.get("name_only")

    assert method.entity_key("  Acme   Controller ", "DEVICE") == method.entity_key(
        "acme controller", "PART"
    )
    assert method.make_entity({"type": "DEVICE", "description": "part"}, "Acme") == {
        "name": "Acme",
        "description": "part",
    }
    assert '"type"' not in method.output_shape().split("relationships")[0]


def test_yaml_spec_rejects_implementation_path_traversal(tmp_path: Path) -> None:
    root = tmp_path / "extraction_methods"
    (root / "specifications").mkdir(parents=True)
    (root / "implementations").mkdir()
    (root / "specifications" / "bad.yaml").write_text(
        "id: bad\nname: Bad\nimplementation:\n  file: ../../outside.py\n  class: Bad\n",
        encoding="utf-8",
    )

    try:
        load_extraction_method_specs(root)
    except ValueError as exc:
        assert "implementations/" in str(exc)
    else:
        raise AssertionError("path traversal must be rejected")
    finally:
        load_extraction_method_specs()
