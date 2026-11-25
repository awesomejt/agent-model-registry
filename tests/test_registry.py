import shutil
from pathlib import Path

import yaml

from model_registry.registry import ROOT, artifact_key, deployment_key, load, short_id


def test_repository_checks_clean():
    reg = load()
    assert reg.errors == []
    assert reg.models and reg.deployments


def test_artifact_key_includes_qat_quantizer_and_variant():
    a = {"format": "gguf", "qat": True, "quant": "UD-Q4_K_XL", "quantizer": "unsloth"}
    assert artifact_key("gemma4-31b", a) == "gemma4-31b:gguf-qat-ud-q4_k_xl-unsloth"
    b = {"format": "gguf", "quant": "UD-Q4_K_XL", "quantizer": "unsloth", "variant": "mtp"}
    assert artifact_key("qwen3.6-35b-a3b", b) == "qwen3.6-35b-a3b:gguf-ud-q4_k_xl-unsloth-mtp"


def test_deployment_key_carries_quality_settings():
    d = {"artifact": "m:gguf-q4_k_m-unsloth", "machine": "box", "devices": ["gpu0", "gpu1"],
         "engine": {"name": "llamacpp", "build": "b123", "backend": "vulkan"},
         "settings": {"kv_cache": "q8_0", "spec": {"type": "draft-mtp", "n": 3}},
         "template": {"name": "effort-v2"}}
    assert deployment_key(d) == "m:gguf-q4_k_m-unsloth@box-gpu0.gpu1/llamacpp-b123-vulkan-draft-mtp3-kvq8-tpl-effort-v2"
    d["settings"]["kv_cache"] = "f16"
    assert "kvq8" not in deployment_key(d)


def test_short_id_is_ref_safe_and_stable():
    sid = short_id("a:b@c/d")
    assert sid == short_id("a:b@c/d") and len(sid) == 8 and sid.isalnum()


def test_resolve_by_alias_key_and_short_id():
    reg = load()
    d = reg.resolve("qwen38-27b")
    key = deployment_key(d)
    assert reg.resolve(key) is d and reg.resolve(short_id(key)) is d
    snap = reg.snapshot(key)
    assert snap["model"]["family"] == "qwen" and snap["artifact"]["quant"] == "UD-Q4_K_XL"


def _copy(tmp_path):
    root = tmp_path / "reg"
    shutil.copytree(ROOT, root, ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", "tests"))
    return root


def test_wrong_declared_key_is_an_error(tmp_path):
    root = _copy(tmp_path)
    p = root / "models/gemma/gemma4-31b.yaml"
    m = yaml.safe_load(p.read_text())
    m["artifacts"][0]["key"] = "gemma4-31b:gguf-q4"
    p.write_text(yaml.safe_dump(m))
    errs = load(root).errors
    assert any("should be 'gemma4-31b:gguf-qat-ud-q4_k_xl-unsloth'" in e for e in errs)


def test_unknown_vocab_and_duplicate_alias_target(tmp_path):
    root = _copy(tmp_path)
    p = root / "models/glm/glm4.7-flash.yaml"
    m = yaml.safe_load(p.read_text())
    m["family"] = "zhipu-glm"
    p.write_text(yaml.safe_dump(m))
    a = root / "aliases.yaml"
    al = yaml.safe_load(a.read_text())
    al["aliases"]["ghost"] = {"deployment": "nope@x/y"}
    a.write_text(yaml.safe_dump(al))
    errs = load(root).errors
    assert any("family 'zhipu-glm'" in e for e in errs)
    assert any("ghost points at unknown deployment" in e for e in errs)
