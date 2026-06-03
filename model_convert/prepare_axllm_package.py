#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover - reported at runtime with a clearer message.
    safe_open = None


EMBED_KEYS = (
    "model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "model.language_model.embed_tokens.weight",
    "model.text_model.embed_tokens.weight",
    "embed_tokens.weight",
)


DEFAULT_POST_CONFIG = {
    "enable_repetition_penalty": False,
    "enable_temperature": False,
    "enable_top_k_sampling": False,
    "enable_top_p_sampling": False,
    "penalty_window": 20,
    "repetition_penalty": 1.2,
    "temperature": 0.9,
    "top_k": 10,
    "top_p": 0.8,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def float32_to_bf16_uint16(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    bits = arr.view(np.uint32)
    rounding_bias = ((bits >> 16) & 1) + 0x7FFF
    return ((bits + rounding_bias) >> 16).astype(np.uint16)


def find_safetensor_with_key(input_path: Path, key: str) -> Path | None:
    if safe_open is None:
        raise RuntimeError("safetensors is required to read safetensors checkpoints")

    index_path = input_path / "model.safetensors.index.json"
    if index_path.exists():
        index = load_json(index_path)
        weight_map = index.get("weight_map", {})
        if key in weight_map:
            return input_path / weight_map[key]

    for path in sorted(input_path.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            if key in f.keys():
                return path
    return None


def load_embedding(input_path: Path) -> torch.Tensor:
    if (input_path / "pytorch_model.bin").exists():
        state = torch.load(input_path / "pytorch_model.bin", map_location="cpu", mmap=True)
        for key in EMBED_KEYS:
            if key in state:
                print(f"found {key} in pytorch_model.bin")
                return state[key].detach().to(torch.float32)

    for key in EMBED_KEYS:
        sf_path = find_safetensor_with_key(input_path, key)
        if sf_path is None:
            continue
        print(f"found {key} in {sf_path}")
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            return f.get_tensor(key).detach().to(torch.float32)

    raise RuntimeError(f"cannot find embedding weight in {input_path}")


def export_embedding(input_path: Path, output_path: Path, filename: str) -> tuple[int, int]:
    tensor = load_embedding(input_path)
    if tensor.ndim != 2:
        raise RuntimeError(f"expected 2-D embedding weight, got shape {tuple(tensor.shape)}")

    output_file = output_path / filename
    arr = tensor.cpu().numpy()
    bf16 = float32_to_bf16_uint16(arr)
    bf16.tofile(output_file)
    print(f"wrote {output_file} shape={tuple(arr.shape)} dtype=bf16")
    return int(arr.shape[0]), int(arr.shape[1])


def load_tokenizer_exporter(converter_path: Path):
    if not converter_path.exists():
        raise RuntimeError(f"tokenizer converter not found: {converter_path}")
    spec = importlib.util.spec_from_file_location("axera_convert_tokenizer", converter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import tokenizer converter: {converter_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.LlmExporter


def export_tokenizer(input_path: Path, output_path: Path, converter_path: Path) -> None:
    exporter_cls = load_tokenizer_exporter(converter_path)
    args = SimpleNamespace(tokenizer_path=str(input_path), dst_path=str(output_path / "tokenizer.txt"))
    exporter = exporter_cls(args)
    # convert_tokenizer.py has Gemma3-specific SentencePiece accounting, but the
    # shared script does not infer this field by itself.
    exporter.model_type = "gemma3"
    exported = exporter.export_tokenizer()
    print(f"wrote {exported}")


def expected_layer_template(model_type: str, prefill_step_size: int) -> str:
    if prefill_step_size <= 1:
        return f"{model_type}_l%d.axmodel"
    return f"{model_type}_p{prefill_step_size}_l%d_together.axmodel"


def write_axllm_config(
    hf_config: dict,
    output_path: Path,
    model_name: str,
    prefill_step_size: int,
    embed_filename: str,
    tokens_embed_num: int,
    tokens_embed_size: int,
    dynamic_load: bool,
    dynamic_pool: int,
    server_max_output_tokens: int,
) -> None:
    model_type = hf_config.get("model_type", "gemma3_text")
    config = {
        "model_name": model_name,
        "template_filename_axmodel": expected_layer_template(model_type, prefill_step_size),
        "filename_post_axmodel": f"{model_type}_post.axmodel",
        "url_tokenizer_model": "tokenizer.txt",
        "tokenizer_type": "Gemma3",
        "filename_tokens_embed": embed_filename,
        "post_config_path": "post_config.json",
        "axmodel_num": int(hf_config["num_hidden_layers"]),
        "tokens_embed_num": tokens_embed_num,
        "tokens_embed_size": tokens_embed_size,
        "pad_token_id": int(hf_config.get("pad_token_id", 0)),
        "rms_norm_eps": float(hf_config.get("rms_norm_eps", 1e-6)),
        "sliding_window": int(hf_config.get("sliding_window", 0) or 0),
        "layer_types": hf_config.get("layer_types", []),
        "b_use_mmap_load_embed": True,
        "b_use_mmap_load_layer": True,
        "dynamic_load_enable": dynamic_load,
        "dynamic_load_pool_size": dynamic_pool,
        "bos": True,
        "eos": False,
        "server_default_max_tokens": server_max_output_tokens,
        "server_max_output_tokens": server_max_output_tokens,
    }
    dump_json(output_path / "config.json", config)
    dump_json(output_path / "post_config.json", DEFAULT_POST_CONFIG)
    print(f"wrote {output_path / 'config.json'}")
    print(f"wrote {output_path / 'post_config.json'}")


def check_expected_axmodels(output_path: Path, model_type: str, layers: int, prefill_step_size: int, strict: bool) -> None:
    missing = []
    template = expected_layer_template(model_type, prefill_step_size)
    for layer in range(layers):
        path = output_path / (template % layer)
        if not path.exists():
            missing.append(path.name)
    post = output_path / f"{model_type}_post.axmodel"
    if not post.exists():
        missing.append(post.name)

    if not missing:
        print("all expected axmodels exist")
        return

    msg = "missing expected axmodel files: " + ", ".join(missing[:8])
    if len(missing) > 8:
        msg += f", ... ({len(missing)} total)"
    if strict:
        raise RuntimeError(msg)
    print(f"warning: {msg}")


def main() -> None:
    root = repo_root()
    default_input_path = root.parent / "models" / "gemma-3-270m-it"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-path", default=str(default_input_path))
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--prefill-step-size", type=int, default=128)
    parser.add_argument("--embed-filename", default="model.embed_tokens.weight.bfloat16.bin")
    parser.add_argument(
        "--tokenizer-converter",
        default=str(root / "ax-llm/third_party/tokenizer.axera/tests/convert_tokenizer.py"),
    )
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--strict-axmodels", action="store_true")
    parser.add_argument("--dynamic-load", action="store_true")
    parser.add_argument("--dynamic-pool", type=int, default=2)
    parser.add_argument("--server-max-output-tokens", type=int, default=256)
    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name or f"AXERA-TECH/{input_path.name}"

    hf_config = load_json(input_path / "config.json")
    if hf_config.get("model_type") != "gemma3_text":
        raise RuntimeError(f"expected model_type=gemma3_text, got {hf_config.get('model_type')!r}")

    tokens_embed_num = int(hf_config["vocab_size"])
    tokens_embed_size = int(hf_config["hidden_size"])

    if not args.skip_embedding:
        tokens_embed_num, tokens_embed_size = export_embedding(input_path, output_path, args.embed_filename)

    if not args.skip_tokenizer:
        export_tokenizer(input_path, output_path, Path(args.tokenizer_converter).resolve())

    write_axllm_config(
        hf_config=hf_config,
        output_path=output_path,
        model_name=model_name,
        prefill_step_size=args.prefill_step_size,
        embed_filename=args.embed_filename,
        tokens_embed_num=tokens_embed_num,
        tokens_embed_size=tokens_embed_size,
        dynamic_load=args.dynamic_load,
        dynamic_pool=args.dynamic_pool,
        server_max_output_tokens=args.server_max_output_tokens,
    )
    check_expected_axmodels(
        output_path=output_path,
        model_type=hf_config.get("model_type", "gemma3_text"),
        layers=int(hf_config["num_hidden_layers"]),
        prefill_step_size=args.prefill_step_size,
        strict=args.strict_axmodels,
    )


if __name__ == "__main__":
    main()
