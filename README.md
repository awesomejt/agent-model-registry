# agent-model-registry

Local AI database: a registry of local LLMs, the weights files (artifacts) made from them, and how each is deployed
(engine, backend, machine, settings). Every entry has a unique key **derived from its attributes**, so a key always
says what the entry is, and benchmark results from any tool can be grouped by family, architecture, size, quant,
engine or machine. Plain YAML in git is the first version of the database. Loading it into SQLite or DuckDB later
takes one script.

Used by modelbench (quality), and intended for the speed and context tools in `local/benchmarks`.

```bash
uv sync
uv run registry check                    # validate everything (errors exit 1; warnings list unverified facts)
uv run registry list deployments --where family=qwen --where architecture=moe
uv run registry list artifacts --where quant=UD-Q4_K_XL
uv run registry show qwen38-27b          # resolved model + artifact + deployment + machine as JSON
uv run pytest -q
```

## Layers and keys

| layer | file | key | example |
|---|---|---|---|
| model | `models/<family>/<key>.yaml` | vendor name, lowercase; MoE carries `-a<active>b` | `qwen3.8-27b`, `qwen3.6-35b-a3b`, `gemma4-26b-a4b` |
| artifact | inside the model file | `<model>:<format>[-qat]-<quant>-<quantizer>[-<variant>]` | `gemma4-31b:gguf-qat-ud-q4_k_xl-unsloth` |
| machine | `machines/<key>.yaml` | short host name | `aiserver`, `macmini`, `workstation` |
| deployment | inside the machine file | `<artifact>@<machine>[-<devices>]/<engine>[-<build>][-<backend>][-<flags>]` | `qwen3.8-27b:gguf-ud-q4_k_xl-unsloth@aiserver-gpu0/llamacpp-29b52d398-vulkan-draft-mtp3-kvq8` |
| alias | `aliases.yaml` | anything short | `qwen38-27b` → a deployment, or a `route` over several |

Flags, in order: speculative decoding (`draft-mtp3`, `dflash2`), KV-cache type when not f16/bf16 (`kvq8`,
`kvaffine8`), a non-default chat template (`tpl-<name>`), and an optional `tag` for deployments that would otherwise
share a key (for example the same everything at two context sizes).

Rules:
- **Keys are derived.** `check` recomputes every artifact and deployment key from its fields and rejects a mismatch.
  To add an entry, fill in the fields, run `check`, and copy the key it prints.
- **Deployment keys never change.** An engine upgrade, a different KV cache, a different template or a different
  quant means a *new* deployment entry. Results that reference the old key stay correct. Repoint the alias
  instead.
- **Short id**: `registry list deployments` prints an 8-hex id (sha256 of the key). It is safe in git branch names
  and file names, which can't contain `:` or `@`.
- **Fine-tunes** keep the base family and vendor (so grouping and same-vendor judge exclusion work), and set
  `base_model` and `publisher`.
- **Unverified facts** stay `null` or are listed in `unverified:`. `check` reports them as warnings. Never guess a
  parameter count or license.
- **Controlled values** (families, architectures, formats, quantizers, engines, backends) live in `vocab.yaml`. Add
  a value there first.

## What does not belong here

This repository is public. Keep out LAN or Tailscale addresses, hostnames beyond the short machine key, API keys,
benchmark items, and benchmark scores. Endpoints stay in each tool's own config (for example modelbench
`config/models.yaml`), and results stay in the tools' own result repositories, which reference deployment keys from
here.

## Run records

A tool that runs a deployment should store `registry show <name>` output (the resolved snapshot) with its results, so
results stay self-contained if an entry is corrected later. Where the engine reports it (llama.cpp `/props`), the
tool should compare the served model file and build against the snapshot and warn on drift (an alias quietly
repointed at new weights).

Per-run settings that are not part of a deployment (harness, reasoning effort, sampling) belong to the run, written
after the key as `+<harness>-<effort>` when a single string is needed.
