# Installing LeFlur

LeFlur ships as part of the `lbster` package on PyPI, but pulls in heavy
optional dependencies (Latent Generator + bionemo-moco + ESMFold) only when
you explicitly request them. Pick the install profile that matches your
hardware.

## 1. Install Python dependencies

We use [`uv`](https://github.com/astral-sh/uv) for fast, reproducible
installs. From a fresh clone:

```bash
git clone https://github.com/prescient-design/lobster.git
cd lobster
```

Then pick one of the two install profiles:

### GPU machine (recommended for inference)

```bash
uv sync --extra mgm --extra struct-gpu
```

This pulls flash-attention, CUDA-built PyTorch Geometric, and the GPU
build of Latent Generator. Expect ~5 minutes on a warm cache.

### CPU-only machine (good for testing / smoke runs)

```bash
uv sync --extra mgm --extra struct-cpu
```

ESMFold and most generation modes still run on CPU, but throughput is
~50–100× lower than GPU. Useful for unit tests, the autoencode pipeline,
and Tier-1 dispatch smoke tests.

### Activating the environment

You can either source the venv...

```bash
source .venv/bin/activate
lobster_generate --help
```

...or wrap every command with `uv run`:

```bash
uv run lobster_generate --help
```

## 2. Configure environment variables

LeFlur honours four environment variables. All are optional with sensible
defaults; set them once in your shell profile if you want non-default
locations.

| Variable | Default | What it controls |
|---|---|---|
| `LOBSTER_CACHE` | `~/.cache/lobster/leflur` | Cache root for downloaded checkpoints, benchmark fixtures, and the Foldseek binary. ~30 GiB if you fetch everything. |
| `LOBSTER_OUT` | `~/leflur_out` | Default root for generated artifacts (`output_dir` falls under this). |
| `HF_TOKEN` | (none) | HuggingFace authentication token. Public LeFlur checkpoints work without one, but providing it raises rate limits and is required if the repo is ever made private. |
| `FOLDSEEK_BIN` | `${LOBSTER_CACHE}/foldseek/bin` | Optional Foldseek binary for structural diversity metrics. Skip if `generation.calculate_foldseek_diversity=false`. |

## 3. Authenticate with HuggingFace

The three canonical checkpoints live on a public repo
([`Sidney-Lisanza/leflur`](https://huggingface.co/Sidney-Lisanza/leflur)),
so any valid token works:

```bash
# Either export it directly...
export HF_TOKEN=hf_xxx

# ...or use the HuggingFace CLI (writes ~/.cache/huggingface/token)
uv run huggingface-cli login
```

Generate a token at https://huggingface.co/settings/tokens (read scope is
sufficient).

## 4. (Optional) Install Foldseek for structural diversity metrics

`lobster_generate` reports structural diversity via Foldseek clustering by
default for unconditional generation. If you skip Foldseek, set
`generation.calculate_foldseek_diversity=false` on the CLI.

To install Foldseek:

```bash
# Linux x86_64
mkdir -p "${LOBSTER_CACHE:-$HOME/.cache/lobster/leflur}/foldseek" && \
  curl -fsSL https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz \
    | tar -xz -C "${LOBSTER_CACHE:-$HOME/.cache/lobster/leflur}/foldseek" --strip-components=1
export FOLDSEEK_BIN="${LOBSTER_CACHE:-$HOME/.cache/lobster/leflur}/foldseek/bin/foldseek"
```

Other platforms: see [the Foldseek README](https://github.com/steineggerlab/foldseek).

## 5. Verify the install

```bash
uv run lobster_leflur_checkpoints list
```

You should see three rows: `leflur-base`, `leflur-ted`, and `leflur-pl`. If
this command fails, the most common cause is a missing optional
dependency — re-run `uv sync` with the appropriate `--extra` flags.

## Next steps

- Run your first generation: [`quickstart.md`](quickstart.md)
- Understand the three canonical checkpoints: [`checkpoints.md`](checkpoints.md)
- Full CLI reference: [`cli.md`](cli.md)

## Troubleshooting

**`HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'`**
Your `HF_TOKEN` is unset or invalid. Re-export it or run
`huggingface-cli login` again.

**`OSError: libtorch_cuda.so: cannot open shared object file`**
You installed `--extra struct-cpu` on a GPU machine. Re-run with
`--extra struct-gpu`.

**`flash_attn` build errors during `uv sync --extra mgm`**
Flash-attention requires CUDA at install time. Either build it from
source (see [the flash-attention README](https://github.com/Dao-AILab/flash-attention))
or drop the `mgm` extra — LeFlur generation does not require flash-attention.

**Checkpoint downloads hang or fail with 401**
Verify `HF_TOKEN` has at least read scope on the public
`Sidney-Lisanza/leflur` repo. As a one-off, you can also pre-download a
specific checkpoint outside the CLI:

```bash
uv run lobster_leflur_checkpoints fetch leflur-ted
```
