"""Load, validate and resolve the model registry.

Layers (see README.md): model -> artifact -> deployment, plus aliases that point at deployments.
Artifact and deployment keys are derived from their attributes; `check` refuses an entry whose declared key differs
from the derived one, so a key always says what the entry is and two different entries cannot share a key."""
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODEL_KEY = re.compile(r"^[a-z0-9][a-z0-9.]*(-[a-z0-9.]+)*$")
SLUG_BAD = re.compile(r"[^a-z0-9._]+")


def slug(s):
    return SLUG_BAD.sub("-", str(s).lower()).strip("-")


def artifact_key(model_key, a):
    """`<model>:<format>[-qat]-<quant>-<quantizer>[-<variant>]`, e.g. `gemma4-31b:gguf-qat-ud-q4_k_xl-unsloth`.
    `variant` separates two files with the same quant from the same quantizer (e.g. one carrying MTP tensors)."""
    parts = ([a["format"]] + (["qat"] if a.get("qat") else []) + [a["quant"], a["quantizer"]]
             + ([a["variant"]] if a.get("variant") else []))
    return f"{model_key}:" + "-".join(slug(p) for p in parts)


def deployment_flags(d):
    """Quality-affecting settings that belong in the key, in a fixed order."""
    s = d.get("settings") or {}
    flags = []
    spec = s.get("spec")
    if spec:
        flags.append(slug(f"{spec['type']}{spec.get('n', '')}"))
    kv = s.get("kv_cache")
    if kv and kv not in ("f16", "bf16"):
        flags.append("kv" + slug(kv).replace("_0", "").replace("_", ""))
    tpl = (d.get("template") or {}).get("name")
    if tpl and tpl != "default":
        flags.append("tpl-" + slug(tpl))
    if d.get("tag"):
        flags.append(slug(d["tag"]))
    return flags


def deployment_key(d):
    """`<artifact>@<machine>-<devices>/<engine>-<build>-<backend>[-<flags>]`."""
    e = d["engine"]
    eng = [e["name"]] + ([e["build"]] if e.get("build") else []) + ([e["backend"]] if e.get("backend") else [])
    where = d["machine"] + ("-" + ".".join(slug(x) for x in d["devices"]) if d.get("devices") else "")
    return f"{d['artifact']}@{where}/" + "-".join([slug(x) for x in eng] + deployment_flags(d))


def short_id(key):
    """Stable, git-ref-safe handle for a key (keys contain `:` and `@`, which branch names cannot)."""
    return hashlib.sha256(key.encode()).hexdigest()[:8]


@dataclass
class Registry:
    vocab: dict
    models: dict = field(default_factory=dict)       # key -> model dict (without artifacts)
    artifacts: dict = field(default_factory=dict)    # key -> artifact dict (with "model")
    machines: dict = field(default_factory=dict)     # key -> machine dict (without deployments)
    deployments: dict = field(default_factory=dict)  # key -> deployment dict
    aliases: dict = field(default_factory=dict)      # name -> {"deployment": key} or {"route": [...]}
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    sources: dict = field(default_factory=dict)      # key -> file it came from

    def resolve(self, name):
        """A deployment from its full key, short id, or alias; aliases that are routes return None."""
        if name in self.deployments:
            return self.deployments[name]
        for k, d in self.deployments.items():
            if short_id(k) == name:
                return d
        a = self.aliases.get(name)
        if a and a.get("deployment"):
            return self.deployments.get(a["deployment"])
        return None

    def snapshot(self, key):
        """Everything known about a deployment, flattened for embedding in a run's meta.json."""
        d = self.deployments[key]
        a = self.artifacts[d["artifact"]]
        m = self.models[a["model"]]
        mach = self.machines[d["machine"]]
        return {"key": key, "id": short_id(key), "deployment": d, "artifact": a, "model": m,
                "machine": {k: v for k, v in mach.items() if k != "deployments"}}


def _load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load(root=ROOT):
    root = Path(root)
    reg = Registry(vocab=_load_yaml(root / "vocab.yaml"))
    v = reg.vocab

    def add(table, key, value, src, what):
        if key in table:
            reg.errors.append(f"{src}: duplicate {what} key {key} (also in {reg.sources[key]})")
            return
        table[key] = value
        reg.sources[key] = src

    def vocab_check(src, what, value, vocab_name):
        if value is not None and value not in v.get(vocab_name, []):
            reg.errors.append(f"{src}: {what} {value!r} not in vocab.yaml {vocab_name}")

    for path in sorted((root / "models").rglob("*.yaml")):
        src = str(path.relative_to(root))
        m = _load_yaml(path)
        key = m.get("key", "")
        if not MODEL_KEY.match(key):
            reg.errors.append(f"{src}: model key {key!r} must be lowercase letters, digits, dots and dashes")
        if path.stem != key:
            reg.errors.append(f"{src}: file name must be the model key ({key}.yaml)")
        for f in v.get("model_required", []):
            if f not in m:
                reg.errors.append(f"{src}: model missing field {f}")
        vocab_check(src, "family", m.get("family"), "families")
        vocab_check(src, "architecture", m.get("architecture"), "architectures")
        if m.get("architecture") == "dense" and m.get("params_active_b") not in (None, m.get("params_total_b")):
            reg.errors.append(f"{src}: dense model with params_active_b != params_total_b")
        if (m.get("architecture") in ("moe", "hybrid-moe") and m.get("params_active_b") is not None
                and not m.get("vendor_name_key")):
            act = m["params_active_b"]
            forms = {f"{act:g}", f"{round(act):d}"}  # A3B is the usual vendor spelling of 3.3B active
            if not any(re.search(rf"-a{re.escape(f)}b(-|$)", key) for f in forms):
                reg.warnings.append(f"{src}: MoE key {key} should carry -a{round(act):d}b")
        for f in m.get("unverified", []):
            reg.warnings.append(f"{src}: unverified {f}")
        arts = m.pop("artifacts", []) or []
        add(reg.models, key, m, src, "model")
        for a in arts:
            a = dict(a, model=key)
            vocab_check(src, "format", a.get("format"), "formats")
            vocab_check(src, "quantizer", a.get("quantizer"), "quantizers")
            want = artifact_key(key, a)
            if a.get("key") != want:
                reg.errors.append(f"{src}: artifact key {a.get('key')!r} should be {want!r}")
            for f in a.get("unverified", []):
                reg.warnings.append(f"{src}: {want}: unverified {f}")
            add(reg.artifacts, want, a, src, "artifact")

    for path in sorted((root / "machines").glob("*.yaml")):
        src = str(path.relative_to(root))
        mach = _load_yaml(path)
        mkey = mach.get("key")
        if path.stem != mkey:
            reg.errors.append(f"{src}: file name must be the machine key ({mkey}.yaml)")
        deps = mach.pop("deployments", []) or []
        add(reg.machines, mkey, mach, src, "machine")
        for d in deps:
            d = dict(d, machine=mkey)
            vocab_check(src, "engine", d.get("engine", {}).get("name"), "engines")
            vocab_check(src, "backend", d.get("engine", {}).get("backend"), "backends")
            if d.get("artifact") not in reg.artifacts:
                reg.errors.append(f"{src}: deployment refers to unknown artifact {d.get('artifact')!r}")
            for dev in d.get("devices", []):
                if dev not in (mach.get("devices") or {}):
                    reg.errors.append(f"{src}: unknown device {dev!r} on {mkey}")
            want = deployment_key(d)
            if d.get("key") != want:
                reg.errors.append(f"{src}: deployment key {d.get('key')!r} should be {want!r}")
            for f in d.get("unverified", []):
                reg.warnings.append(f"{src}: {want}: unverified {f}")
            add(reg.deployments, want, d, src, "deployment")

    alias_file = root / "aliases.yaml"
    if alias_file.exists():
        for name, a in (_load_yaml(alias_file).get("aliases") or {}).items():
            reg.aliases[name] = a
            targets = [a["deployment"]] if a.get("deployment") else list(a.get("route") or [])
            if not targets:
                reg.errors.append(f"aliases.yaml: {name} needs `deployment` or `route`")
            for t in targets:
                if t not in reg.deployments:
                    reg.errors.append(f"aliases.yaml: {name} points at unknown deployment {t!r}")
            if name in reg.deployments:
                reg.errors.append(f"aliases.yaml: alias {name} collides with a deployment key")

    ids = {}
    for k in reg.deployments:
        sid = short_id(k)
        if sid in ids:
            reg.errors.append(f"short id {sid} collides: {k} and {ids[sid]} (lengthen short_id)")
        ids[sid] = k
    return reg
