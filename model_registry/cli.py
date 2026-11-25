"""`registry` command: check, list, show, key."""
import argparse
import json
import sys

import yaml

from .registry import ROOT, artifact_key, deployment_key, load, short_id


def main(argv=None):
    ap = argparse.ArgumentParser(prog="registry", description="Local model registry")
    ap.add_argument("--root", default=str(ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="validate every file; exit 1 on errors")
    c.add_argument("--quiet", action="store_true", help="hide warnings")
    ls = sub.add_parser("list", help="list entries")
    ls.add_argument("what", choices=["models", "artifacts", "deployments", "aliases"])
    ls.add_argument("--where", action="append", default=[], metavar="FIELD=VALUE",
                    help="filter, e.g. family=qwen, architecture=moe, quant=UD-Q4_K_XL, engine=llama.cpp")
    sh = sub.add_parser("show", help="resolved snapshot of a deployment (key, short id or alias) as JSON")
    sh.add_argument("name")
    k = sub.add_parser("key", help="print the derived key for an artifact or deployment given as YAML on stdin")
    k.add_argument("--model", help="model key, for an artifact")
    a = ap.parse_args(argv)
    reg = load(a.root)

    if a.cmd == "check":
        for w in ([] if a.quiet else reg.warnings):
            print("warning:", w)
        for e in reg.errors:
            print("error:", e)
        print(f"{len(reg.models)} models, {len(reg.artifacts)} artifacts, {len(reg.machines)} machines, "
              f"{len(reg.deployments)} deployments, {len(reg.aliases)} aliases; "
              f"{len(reg.errors)} errors, {len(reg.warnings)} warnings")
        return 1 if reg.errors else 0

    if a.cmd == "list":
        conds = [w.split("=", 1) for w in a.where]

        def flat(key):
            if a.what == "models":
                return reg.models[key]
            if a.what == "artifacts":
                art = reg.artifacts[key]
                return {**reg.models[art["model"]], **art}
            d = reg.deployments[key]
            art = reg.artifacts[d["artifact"]]
            return {**reg.models[art["model"]], **art, **d, "engine": d["engine"]["name"],
                    "backend": d["engine"].get("backend"), "id": short_id(key)}

        if a.what == "aliases":
            for n, t in sorted(reg.aliases.items()):
                print(f"{n:24} -> {t.get('deployment') or 'route ' + ', '.join(t['route'])}")
            return 0
        table = {"models": reg.models, "artifacts": reg.artifacts, "deployments": reg.deployments}[a.what]
        for key in sorted(table):
            row = flat(key)
            if all(str(row.get(f)) == v for f, v in conds):
                print(f"{short_id(key)}  {key}" if a.what == "deployments" else key)
        return 0

    if a.cmd == "show":
        d = reg.resolve(a.name)
        if not d:
            print(f"unknown or route-only name: {a.name}", file=sys.stderr)
            return 1
        print(json.dumps(reg.snapshot(deployment_key(d)), indent=2, default=str))
        return 0

    if a.cmd == "key":
        entry = yaml.safe_load(sys.stdin)
        print(artifact_key(a.model, entry) if a.model else deployment_key(entry))
        return 0


if __name__ == "__main__":
    sys.exit(main())
