# gemma-3-270m-it AXERA Conversion

This directory contains the conversion flow for the local HuggingFace model:

```text
../models/gemma-3-270m-it
```

Gemma-3-270m-it is a text-only LLM, so the flow is simpler than `Qwen3-VL.AXERA`: there is no vision encoder export. The conversion produces LLM axmodels plus the files required by `ax-llm`.

## Environment

From the repository root:

```bash
cd npu-codebase
uv sync --extra cpu
source script/npu_dev
build_cmodel -b ax650npu
cd ..
```

The scripts also try to source `npu-codebase/script/npu_dev` automatically when `pulsar2` is not already in `PATH`.

## Build

Default build:

```bash
cd gemma-3-270m.axera/model_convert
bash build_all.sh
```

Defaults:

```text
INPUT_DIR=${REPO_ROOT}/../models/gemma-3-270m-it
PREFILL_STEP_SIZE=128
PREFILL_LEN=1024
MAX_CONTEXT=2048
CHIP=AX650
PARALLEL=8
WEIGHT_TYPE=s8
POST_WEIGHT_TYPE=s8
OUTPUT_DIR=../gemma-3-270m-it-AX650-C128-P1024-CTX2047
```

`MAX_CONTEXT` is passed to `pulsar2 llm_build2 --max_context`. The actual maximum
KV cache length is `MAX_CONTEXT - 1`, so the default package name uses `CTX2047`.
`INPUT_DIR` defaults to `${REPO_ROOT}/../models/gemma-3-270m-it`, where `REPO_ROOT`
is the repository root. The model package name is derived from `INPUT_DIR` by
default.

Override any value through environment variables:

```bash
PREFILL_LEN=1152 MAX_CONTEXT=2048 PARALLEL=4 bash build_all.sh
```

`build_all.sh` runs:

1. `pulsar2 llm_build2` to compile decoder/prefill/post axmodels.
2. `prepare_axllm_package.py` to export:
  - `model.embed_tokens.weight.bfloat16.bin`
   - `tokenizer.txt`
   - `config.json`
   - `post_config.json`

## Prepare Package Only

If axmodels were already compiled:

```bash
cd gemma-3-270m.axera/model_convert
OUTPUT_DIR=/path/to/gemma-3-270m-it-AX650-C128-P1024-CTX2047 bash prepare_package.sh
```

Use `STRICT_AXMODELS=1` to fail if expected axmodel files are missing.

## Run With ax-llm

After copying the generated model package to the target device:

```bash
axllm run /path/to/gemma-3-270m-it-AX650-C128-P1024-CTX2047
```

or:

```bash
axllm serve /path/to/gemma-3-270m-it-AX650-C128-P1024-CTX2047 --port 8000
```

The generated `config.json` sets:

```json
{
  "tokenizer_type": "Gemma3",
  "sliding_window": 512,
  "layer_types": ["sliding_attention", "..."]
}
```

`Gemma3` uses the Gemma turn-style chat template from `chat_template.jinja`.
`layer_types` and `sliding_window` are required so `ax-llm` can build the correct
full/sliding attention masks. RoPE local/global differences are handled in the
compiled axmodels by `npu-codebase/yasched/llm_builder/gemma_test.py`.

## Generated Package

The default conversion in this workspace generated:

```text
gemma-3-270m.axera/gemma-3-270m-it-AX650-C128-P1024-CTX2047/
```

The package contains 18 layer axmodels, `gemma3_text_post.axmodel`, the BF16
embedding table, `tokenizer.txt`, `config.json`, and `post_config.json`.

## Notes

- `gemma-3-270m-it` is instruction tuned, but it is still a very small 270M model. Keep prompts and expected answer quality modest.
- `tools/llm_smoke` in the current `ax-llm` branch does not parse `layer_types/sliding_window`; prefer `axllm run` or `axllm serve` for Gemma3 sliding-attention validation unless that smoke tool is updated.
