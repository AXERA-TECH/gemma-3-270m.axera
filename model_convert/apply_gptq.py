from __future__ import annotations

import io
import math
import os
import random
from itertools import cycle
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Dict, Iterator, List, Sequence

os.environ.setdefault(
    "HF_DATASETS_CACHE",
    os.environ.get("CALIB_CACHE_DIR", "/tmp/gemma3_hf_datasets_cache"),
)

import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_ID = os.environ.get("MODEL_ID", "../../../models/gemma-3-270m-it/")
QUANT_PATH = os.environ.get("QUANT_PATH", "../gemma-3-270m-it-gptq-int4")
MODEL_DEVICE = os.environ.get("MODEL_DEVICE", "cuda:0")
CALIBRATION_DEVICE = os.environ.get("CALIBRATION_DEVICE", MODEL_DEVICE)

NUM_CALIB = int(os.environ.get("NUM_CALIB", "1024"))
NUM_TEXT_CALIB = int(os.environ.get("NUM_TEXT_CALIB", str(NUM_CALIB)))  # Gemma-3 is text-only
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))

WIKI_DATASET_ID = os.environ.get("WIKI_DATASET_ID", "wikimedia/wikipedia")
WIKI_ZH_CONFIG = os.environ.get("WIKI_ZH_CONFIG", "20231101.zh")
WIKI_EN_CONFIG = os.environ.get("WIKI_EN_CONFIG", "20231101.en")
TEXT_LANG_PATTERN = tuple(
    item.strip().lower()
    for item in os.environ.get("TEXT_LANG_PATTERN", "zh,en").split(",")
    if item.strip()
)
SHUFFLE_BUFFER = int(os.environ.get("SHUFFLE_BUFFER", "10000"))
MIN_TEXT_CHARS = int(os.environ.get("MIN_TEXT_CHARS", "128"))
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "1536"))

MAX_CALIB_TOKENS = int(os.environ.get("MAX_CALIB_TOKENS", "2048"))
ADD_GENERATION_PROMPT = os.environ.get("ADD_GENERATION_PROMPT", "1") not in {"0", "false", "False"}
DRY_RUN = os.environ.get("DRY_RUN", "0") in {"1", "true", "True"}

QUANT_MSE = float(os.environ.get("QUANT_MSE", "2.5"))

MODEL_INPUT_KEYS = {
    "input_ids",
    "attention_mask",
}

ZH_SYSTEM_PROMPTS = [
    "你是一个有用的AI助手，请用简体中文回答用户的问题。",
    "请使用中文回答，提供准确且有用的信息。",
    "作为一个AI助手，请用中文详细回答问题。",
]

EN_SYSTEM_PROMPTS = [
    "You are a helpful AI assistant. Please answer the user's questions.",
    "Provide accurate and useful information in your response.",
    "As an AI assistant, please give detailed answers to questions.",
]

def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _select_random_prompt(rng: random.Random, is_chinese: bool) -> str:
    if is_chinese:
        return rng.choice(ZH_SYSTEM_PROMPTS)
    else:
        return rng.choice(EN_SYSTEM_PROMPTS)


def _load_wikipedia_stream(config_name: str, seed: int):
    from datasets import load_dataset

    dataset = load_dataset(WIKI_DATASET_ID, config_name, split="train", streaming=True)
    return dataset.shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER)


def _normalize_text(text: str) -> str:
    compact = " ".join(text.split()).strip()
    if MAX_TEXT_CHARS > 0:
        compact = compact[:MAX_TEXT_CHARS]
    return compact


def _build_text_calibration_dataset(num_samples: int) -> List[Dict[str, Any]]:
    if num_samples <= 0:
        return []

    try:
        streams: Dict[str, Iterator[Dict[str, str]]] = {}
        if "zh" in TEXT_LANG_PATTERN:
            streams["zh"] = iter(_load_wikipedia_stream(WIKI_ZH_CONFIG, seed=42))
        if "en" in TEXT_LANG_PATTERN:
            streams["en"] = iter(_load_wikipedia_stream(WIKI_EN_CONFIG, seed=43))
        if not streams:
            raise ValueError("TEXT_LANG_PATTERN must contain at least one of: zh,en")
    except Exception as exc:
        print(f"warning: skip Wikipedia text calibration because dataset loading failed: {exc}")
        return []

    calibration_dataset: List[Dict[str, Any]] = []
    rng = random.Random(42)

    for language in cycle(TEXT_LANG_PATTERN):
        if len(calibration_dataset) >= num_samples:
            break
        if language not in streams:
            continue
        try:
            row = next(streams[language])
        except StopIteration:
            break
        except Exception as exc:
            print(f"warning: stop Wikipedia text calibration because streaming failed: {exc}")
            break

        text = _normalize_text(str(row.get("text", "")))
        if len(text) < MIN_TEXT_CHARS:
            continue

        is_chinese = language == "zh"
        system_prompt = _select_random_prompt(rng, is_chinese)
        
        calibration_dataset.append({
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": text,
                }
            ],
            "source": f"wikipedia_{language}",
        })

        if len(calibration_dataset) % 64 == 0:
            print(f"collected {len(calibration_dataset)}/{num_samples} text calibration samples")

    print(f"ready text calibration samples: {len(calibration_dataset)}")
    return calibration_dataset


def build_calibration_dataset(num_samples: int) -> List[Dict[str, Any]]:
    text_dataset = _build_text_calibration_dataset(num_samples)
    print(f"ready text calibration samples: {len(text_dataset)}")
    return text_dataset


def _has_assistant_message(messages: Sequence[Dict[str, Any]]) -> bool:
    return any(message.get("role") == "assistant" for message in messages)


def _extract_messages(example: Any) -> List[Dict[str, Any]]:
    if isinstance(example, dict):
        if "messages" in example:
            return example["messages"]
        if "role" in example and "content" in example:
            return [example]
        text = str(example.get("text", ""))
        return [{"role": "user", "content": text}]
    if isinstance(example, list):
        return example
    return [{"role": "user", "content": str(example)}]


def _get_processor(qmodel):
    processor = getattr(qmodel, "tokenizer", None)
    if processor is None and hasattr(qmodel, "load_processor"):
        processor = qmodel.load_processor()
    if processor is None:
        processor = getattr(qmodel, "processor", None)
    if processor is None:
        raise RuntimeError("Gemma-3 calibration requires a tokenizer from GPTQModel or AutoTokenizer.")
    if not hasattr(processor, "apply_chat_template"):
        raise RuntimeError("Gemma-3 calibration processor must support apply_chat_template().")
    return processor


def _prepare_text_calibration_dataset(
    qmodel,
    calibration_dataset,
    calibration_dataset_concat_size=None,
    calibration_dataset_sort="desc",
    batch_size=1,
    calibration_data_min_length=10,
    calibration_concat_separator=None,
):
    del calibration_dataset_concat_size, calibration_concat_separator

    if batch_size != 1:
        print("text-only calibration uses one sample per batch; ignoring batch_size > 1")

    processor = _get_processor(qmodel)
    encoded_batches: List[Dict[str, torch.Tensor]] = []
    skipped_short = 0
    skipped_long = 0
    skipped_error = 0

    for sample_index, example in enumerate(calibration_dataset):
        messages = _extract_messages(example)
        try:
            # Apply chat template for Gemma-3
            encoded = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=ADD_GENERATION_PROMPT and not _has_assistant_message(messages),
                return_dict=True,
                return_tensors="pt",
            )
        except Exception as exc:
            skipped_error += 1
            print(f"warning: skip calibration sample {sample_index} because processor failed: {exc}")
            continue

        input_ids = encoded.get("input_ids")
        if input_ids is None or input_ids.shape[-1] <= calibration_data_min_length:
            skipped_short += 1
            continue
        if MAX_CALIB_TOKENS > 0 and input_ids.shape[-1] > MAX_CALIB_TOKENS:
            skipped_long += 1
            continue

        batch = {
            key: value.detach()
            for key, value in encoded.items()
            if key in MODEL_INPUT_KEYS and torch.is_tensor(value)
        }
        if "attention_mask" not in batch:
            batch["attention_mask"] = torch.ones_like(batch["input_ids"])
        encoded_batches.append(batch)

        if len(encoded_batches) % 64 == 0:
            print(f"encoded {len(encoded_batches)} calibration batches")

    if calibration_dataset_sort == "asc":
        encoded_batches.sort(key=lambda item: item["input_ids"].shape[-1])
    elif calibration_dataset_sort == "desc":
        encoded_batches.sort(key=lambda item: item["input_ids"].shape[-1], reverse=True)
    elif calibration_dataset_sort == "shuffle":
        random.Random(42).shuffle(encoded_batches)

    total_tokens = sum(batch["attention_mask"].sum().item() for batch in encoded_batches if "attention_mask" in batch)
    print(
        f"prepared {len(encoded_batches)} one-sample calibration batches "
        f"({total_tokens} non-padded tokens)"
    )
    if skipped_short or skipped_long or skipped_error:
        print(f"skipped samples: short={skipped_short}, long={skipped_long}, error={skipped_error}")
    if not encoded_batches:
        raise RuntimeError("No usable calibration batches were prepared.")
    return encoded_batches


def _run_dry_prepare(calibration_dataset: List[Dict[str, Any]]) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dummy_model = SimpleNamespace(tokenizer=tokenizer)
    encoded_batches = _prepare_text_calibration_dataset(dummy_model, calibration_dataset, batch_size=1)
    first_batch = encoded_batches[0]
    print("dry run first batch:")
    for key, value in first_batch.items():
        print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")


def main() -> None:
    print(f"model_id={MODEL_ID}")
    print(f"quant_path={QUANT_PATH}")
    print(
        "calibration="
        f"num={NUM_CALIB}, text={NUM_TEXT_CALIB}, "
        f"max_tokens={MAX_CALIB_TOKENS}"
    )

    calibration_dataset = build_calibration_dataset(NUM_CALIB)
    print(f"ready to quantize with {len(calibration_dataset)} text calibration samples")

    if DRY_RUN:
        _run_dry_prepare(calibration_dataset[: min(len(calibration_dataset), 8)])
        return

    from gptqmodel import GPTQModel, QuantizeConfig

    quant_config = QuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False,
        static_groups=True,
        act_group_aware=False,
        sym=True,
        mse=QUANT_MSE,
        calibration_data_device=CALIBRATION_DEVICE,
    )

    model = GPTQModel.load(MODEL_ID, quant_config, device=MODEL_DEVICE)
    model.prepare_dataset = MethodType(_prepare_text_calibration_dataset, model)
    model.quantize(calibration_dataset, batch_size=BATCH_SIZE)
    model.save(QUANT_PATH)


if __name__ == "__main__":
    main()
