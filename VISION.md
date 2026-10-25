# VISION job scripts

## Rules

- Never run `ssh`, `rsync`, or `scp` yourself: each connection triggers Duo.
  Give the user short commands, one per block, labelled **Mac** (fish) or
  **VISION** (Bash). Inside a VISION shell, omit the `ssh vision` wrapper.
- Write placeholders as `<project>`, `<account>`, `<job-id>` and resolve them
  before handing commands over.
- Current limits come from the [Slurm](https://docs.vision.tamus.edu/slurm/),
  [hardware](https://docs.vision.tamus.edu/specifications/), and
  [maintenance](https://docs.vision.tamus.edu/maintenance/) pages; they
  override this file.

## Storage

**All caches must live under `/scratch/user/$USER`, including download,
package, model, and compilation caches. Set their paths before any
installation, download, or Python import; never use home or `/tmp` for
caches.**

| Path | Size | Use |
| --- | --- | --- |
| `/home/$USER` | 10 GB, backed up | Dotfiles and credentials only (`~/.netrc`). Never caches or envs. |
| `/scratch/user/$USER` | 1 TB, not backed up | Shared download caches (`hf/`, `uv-cache/`, `uv-python/`, `.cache/`) |
| `/scratch/user/$USER/<project>` | (same quota) | Code, `.venv`, data, `logs/`, `runs/`, checkpoints |
| `/scratch/project/<account>` | Per project | Shared data; the directory name is the Slurm account |
| `/scratch/user/$USER/.cache/jobs/` | Same scratch quota | Unique per-job/task compilation caches; clean up after workers exit |

Tools default to `~/.cache` and fill home silently, so every setup step and
job sources the env script below first. Cache variables don't redirect
outputs: point datasets, checkpoints and run directories at the project
directory explicitly. Quotas, including file counts, are printed in the login
banner; `df` does not show personal quota. Download verified results: scratch
is not backed up.

## Environment script

Save as `scripts/env.sh` in the project and `source` it in setup and in every
job, before any Python import:

```bash
export VISION_SCRATCH=/scratch/user/$USER
export PROJECT_DIR=$VISION_SCRATCH/<project>
export XDG_CACHE_HOME=$VISION_SCRATCH/.cache
export HF_HOME=$VISION_SCRATCH/hf
export HF_DATASETS_CACHE=$HF_HOME/datasets
export UV_CACHE_DIR=$VISION_SCRATCH/uv-cache
export UV_PYTHON_INSTALL_DIR=$VISION_SCRATCH/uv-python
export UV_PYTHON_PREFERENCE=only-managed
export PIP_CACHE_DIR=$XDG_CACHE_HOME/pip
export TORCH_HOME=$XDG_CACHE_HOME/torch
export WANDB_DIR=$PROJECT_DIR/runs
export WANDB_CACHE_DIR=$XDG_CACHE_HOME/wandb
export WANDB_DATA_DIR=$XDG_CACHE_HOME/wandb-data
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch-extensions"
mkdir -p "$XDG_CACHE_HOME/jobs" "$HF_HOME" "$UV_CACHE_DIR" \
  "$UV_PYTHON_INSTALL_DIR" "$WANDB_DIR"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  task_id=$SLURM_JOB_ID-${SLURM_ARRAY_TASK_ID:-0}
  job_cache=$(mktemp -d "$XDG_CACHE_HOME/jobs/$task_id.XXXXXX")
  export TRITON_CACHE_DIR=$job_cache/triton
  export VLLM_CACHE_ROOT=$job_cache/vllm
  export CUDA_CACHE_PATH=$job_cache/cuda
  export TORCH_EXTENSIONS_DIR=$job_cache/torch-extensions
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
mkdir -p "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$CUDA_CACHE_PATH" \
  "$TORCH_EXTENSIONS_DIR"
```

- Each task sources this script to get its own scratch compilation-cache
  directory. Remove that directory only after its workers exit; keep shared
  download caches for reuse.
- Run `hf auth login` and `wandb login` once on the login node, with this
  script sourced. A token saved under a different `HF_HOME` is invisible to
  jobs.
- Prefetch exact model/dataset revisions on the login node. In offline jobs,
  `HfApi().list_repo_files` and `load_dataset(..., streaming=True)` fail even
  when everything is cached, so use known file paths and non-streaming loads.
  Start each job with a check that the needed files are cached, so it fails
  fast.
- If `wandb.init` fails in a job, use `mode="offline"` and run `wandb sync`
  from the login node.

## Building the environment

`uv` is not on PATH. Install a pinned copy into the project, using [unmanaged
mode](https://docs.astral.sh/uv/reference/installer/#unmanaged-installations)
so it doesn't edit shell profiles:

```bash
curl -LsSf https://astral.sh/uv/0.12.21/install.sh |
  env UV_UNMANAGED_INSTALL=/scratch/user/$USER/<project>/.tools sh
```

```bash
.tools/uv sync --locked
```

- Build on uv-managed CPython, never Ubuntu's system Python 3.12.3: on it,
  torchrun misparsed script flags
  (`ambiguous option: --n could match --nnodes`). If uv reports
  `No interpreter found for Python 3.12.x`, pin a version from
  `.tools/uv python list`.
- `--frozen` skips the lock consistency check; use `--locked` so a stale lock
  fails.
- Install on the login node. Jobs call `.venv/bin/python` directly and never
  run `uv` or `pip`.

## CUDA and wheels

The platform runs Ubuntu 24.04 x86_64 with glibc 2.39, and NVIDIA driver
580.126.16 (`nvidia-smi` shows CUDA 13.0). There is **no `nvcc` and no
`CUDA_HOME`** on compute nodes, so anything that compiles CUDA from source
fails. Each H200 has 139.8 GiB usable as reported by torch.

Use prebuilt wheels whose CUDA build matches torch's build, and pin the wheel
URL and hash:

- vLLM: [release wheels](https://github.com/vllm-project/vllm/releases)
  (`+cu129`, `+cu130` suffixes) and [install
  docs](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html)
- PyTorch: [wheel index](https://download.pytorch.org/whl/torch/) (`+cuXXX`
  local versions)
- FlashAttention (`flash-attn`):
  [release wheels](https://github.com/Dao-AILab/flash-attention/releases). The
  filename must match torch major.minor, CUDA major, `cxx11abi` (TRUE/FALSE,
  matching `torch._C._GLIBCXX_USE_CXX11_ABI`) and the Python tag. With no
  `nvcc`, a pip fallback to a source build fails. FlashAttention and FlashInfer
  are different packages; install only what the workload needs.

Install GPU packages with `--only-binary :all:` so a missing wheel fails
instead of compiling.

| Stack | Pairing that ran |
| --- | --- |
| vLLM inference | `vllm 0.23.0+cu129` ([wheel](https://github.com/vllm-project/vllm/releases/download/v0.23.0/vllm-0.23.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl), sha256 `8bc2203995d061e6b988916b71b9dee8a5970f5fdc5f37d4445a877a2fab2cc1`) + `torch 2.11.0+cu129`, Python 3.12.14, `uv pip install --torch-backend cu129 --only-binary :all:`. Works with Transformers 4.57.6 / Hub 0.36.2 and with Transformers 5.5.1 / Hub 1.11.0. |
| Training (no vLLM) | `torch 2.14.1+cu130` from default PyPI, Python 3.12.12, Transformers 5.18.0, PEFT 0.21.2 |

Never mix CUDA builds in one env: a cu130 torch and a cu129 vLLM are
incompatible.

## Known failures

| Symptom | Cause → fix |
| --- | --- |
| `ImportError: libcudart.so.13` | Plain `pip install vllm` pulled a CUDA 13 wheel against cu129 torch. Install the matching `+cu129` wheel. Never symlink libraries. |
| `Could not find nvcc` from FlashInfer | Its sampler JIT-compiles. Set `VLLM_USE_FLASHINFER_SAMPLER=0` before importing vLLM. Some models also need a Triton backend override. `enforce_eager=True` does **not** stop this JIT. |
| `TypeError: '>' not supported between 'str' and 'int'` in vLLM | In Transformers 5, `apply_chat_template(tokenize=True)` returns a `BatchEncoding`. Pass `return_dict=False`, assert `list[int]`, check that prompt + `max_tokens` ≤ `max_model_len`, and send `{"prompt_token_ids": ids}`. |
| CUDA OOM with two processes on one GPU | Use one full-size process per GPU, plus gradient checkpointing for all-layer training. |
| Bare `wait` returns 0 though a worker died | Store each PID and check every `wait "$pid"`; exit nonzero if any failed. |
| torchrun rendezvous timeout / port clash | Use `--nnodes 1 --master_addr 127.0.0.1 --master_port $((29500 + SLURM_JOB_ID % 1000))`. Don't give your script flags that are prefixes of torchrun's (`--n`, `--no`, `--node`). |
| Progress file says FAILED after resubmitting | It is the old attempt's file. Check the new job in `sacct` and read its own log. |

## Job script

```bash
#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --account=<account>
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=112G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/user/%u/<project>/logs/%x-%j.out
set -euo pipefail
cd /scratch/user/$USER/<project>
source scripts/env.sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
.venv/bin/python - <<'EOF'
import torch
assert torch.cuda.is_available()
print(torch.__version__, torch.version.cuda)
EOF
srun .venv/bin/python -u <entrypoint>
```

- Find the account with `sacctmgr -nP show assoc user=$USER format=Account`.
- `#SBATCH` lines don't expand shell variables; use Slurm patterns (`%u`,
  `%j`, `%x`, `%A_%a`) instead. Create `logs/` before `sbatch`. Command-line
  `sbatch` flags override directives.
- Submit with `sbatch`; running the script with `bash` runs it on the login
  node.
- Policy (Slurm config v3, 2026-09-09): partition `def`, `--time` required,
  48 h and 240 GPUs max. Unset requests default to 14 CPUs/GPU and 8,292 MB/CPU.
  The real flag is `--mem-per-cpu`; the site's `--mempercpu` is a typo.
- Maintenance runs from the 2nd Tuesday of each month at 09:00 CT until
  Wednesday 23:59 CT. A job whose `--time` crosses it stays pending until it
  ends.
- Each VISION DGX node has 8 NVIDIA H200 GPUs.
- For independent experiments that fit on one GPU, use persistent one-GPU
  workers, not a separate Slurm job for every configuration. Each worker loads
  the model once and processes unfinished configurations within the same
  allocation. Use shared configuration locks and resumable checkpoints.
- For eight workers, use `--array=0-7%8 --gres=gpu:1`, with a work loop inside
  each worker. Workers may run across different nodes; don't require a whole
  node unless the workload needs it.
- Leave `CUDA_VISIBLE_DEVICES` as Slurm sets it; with one GPU visible, use
  `cuda:0`. Multi-GPU work needs an explicit launcher (torchrun/srun), and
  that launch path needs its own smoke test.
- GPU jobs only: CPU-only submissions have been warned they may be rejected.
- Smoke-test at toy size through the real entry point, saving and evaluation
  before the full run. Check output counts and content, truncation and peak
  memory. Passing imports or model loading proves nothing.
- Log package versions, `torch.version.cuda`, the driver, visible GPUs and
  launch arguments with each run. A change to code, environment or protocol
  gets a new run ID; never edit code or the env under a running job.

## Monitoring

| Need | Command |
| --- | --- |
| Queue and reason | `squeue --me -o '%.20i %.12T %.10M %.40R'` |
| Final state and exit code | `sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed,NodeList --parsable2` |
| Log path and allocation | `scontrol show job <job-id>` |
| GPU use of a running job | `srun --jobid=<job-id> --overlap nvidia-smi` |
| Priority | `sprio -j <job-id>`, `my-fairshare --brief` |

A job pending with reason `Priority` is queued normally; don't resubmit. Quiet
logs and warnings followed by progress aren't failures. Read the final
exception and the `sacct` exit code. `COMPLETED 0:0` still needs output
validation. Cancel only the specific `<job-id>`, and only when asked.

## Transfer

The Mac's `rsync` is openrsync, which has no `--info=progress2`. Provide the
actual transfer command only, without a dry-run preview unless requested. Use
`--delete` only when asked, and never overwrite code that a running job is
using.

```fish
rsync -az --itemize-changes \
  --exclude '/.venv*/' --exclude '/.tools/' --exclude '/.cache/' \
  --exclude '/data/' --exclude '/runs/' --exclude '/logs/' \
  --exclude '/.git/' --exclude '__pycache__/' \
  <local-repo>/ vision:/scratch/user/serhii_tamu.edu/<project>/
```

```fish
rsync -az --itemize-changes \
  vision:/scratch/user/serhii_tamu.edu/<project>/runs/<run-id>/ \
  <local-repo>/runs/<run-id>/
```

Because `.git` is excluded, write `git rev-parse HEAD` into the run config
before uploading. Add `--exclude '*.safetensors'` to the download unless the
weights are a deliverable.
