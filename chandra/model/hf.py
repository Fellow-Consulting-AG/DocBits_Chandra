from typing import List

from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor, BitsAndBytesConfig

from chandra.model.schema import BatchInputItem, GenerationResult
from chandra.model.util import scale_to_fit
from chandra.prompts import PROMPT_MAPPING
from chandra.settings import settings


def generate_hf(
    batch: List[BatchInputItem],
    model,
    max_output_tokens=None,
    bbox_scale: int = settings.BBOX_SCALE,
    **kwargs,
) -> List[GenerationResult]:
    if max_output_tokens is None:
        max_output_tokens = settings.MAX_OUTPUT_TOKENS

    messages = [
        process_batch_element(item, model.processor, bbox_scale) for item in batch
    ]
    text = model.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, _ = process_vision_info(messages)
    inputs = model.processor(
        text=text,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
        padding_side="left",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_output_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = model.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    results = [
        GenerationResult(raw=out, token_count=len(ids), error=False)
        for out, ids in zip(output_text, generated_ids_trimmed)
    ]
    return results


def process_batch_element(item: BatchInputItem, processor, bbox_scale: int):
    prompt = item.prompt
    prompt_type = item.prompt_type

    if not prompt:
        prompt = PROMPT_MAPPING[prompt_type].replace("{bbox_scale}", str(bbox_scale))

    content = []
    image = scale_to_fit(item.image, max_size=(1536, 1024))  # Further reduce to save GPU memory during inference
    content.append({"type": "image", "image": image})

    content.append({"type": "text", "text": prompt})
    message = {"role": "user", "content": content}
    return message


def load_model():
    import torch

    # Use CPU offloading for large models on limited GPU memory
    device_map = "auto"
    if settings.TORCH_DEVICE:
        device_map = {"": settings.TORCH_DEVICE}

    # Use float32 for numerical stability (float16/bfloat16 cause inf/nan errors)
    dtype = torch.float32

    kwargs = {
        "device_map": device_map,
        "max_memory": {0: "11GiB", "cpu": "60GiB"},  # Leave 4-5GB GPU for inference processing
        "offload_state_dict": True,  # Offload to CPU RAM instead of disk
    }

    # Add 8-bit quantization if enabled (reduces memory usage by ~50%)
    if settings.USE_8BIT_QUANTIZATION and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        kwargs["quantization_config"] = quantization_config
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["dtype"] = dtype

    if settings.TORCH_ATTN:
        kwargs["attn_implementation"] = settings.TORCH_ATTN

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        settings.MODEL_CHECKPOINT, **kwargs
    )
    model = model.eval()
    processor = Qwen3VLProcessor.from_pretrained(settings.MODEL_CHECKPOINT)
    model.processor = processor
    return model
