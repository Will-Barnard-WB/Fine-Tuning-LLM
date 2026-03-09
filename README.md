# Fine-Tuning Llama 2 with QLoRA

Fine-tune Meta's **Llama-2-7b-chat** on a custom instruction dataset using parameter-efficient fine-tuning (PEFT) with 4-bit quantization — runnable on a single GPU.

---

## Overview

This project demonstrates how to fine-tune a large language model efficiently without needing expensive hardware. It uses **QLoRA** (Quantized Low-Rank Adaptation) to reduce memory usage dramatically while preserving model quality.

**Base model:** `NousResearch/Llama-2-7b-chat-hf`  
**Dataset:** `mlabonne/guanaco-llama2-1k`  
**Output model:** pushed to Hugging Face Hub

---

## Key Techniques

- **4-bit quantization** via `bitsandbytes` (NF4 format) — cuts VRAM usage by ~75%
- **LoRA adapters** (`r=64`, `alpha=16`) — trains <1% of parameters
- **Gradient checkpointing** + `paged_adamw_32bit` optimizer for memory efficiency
- **SFTTrainer** from `trl` for streamlined supervised fine-tuning

---

## Stack

`transformers` · `peft` · `trl` · `bitsandbytes` · `accelerate` · `PyTorch`

---

## Usage

```bash
pip install -q accelerate peft bitsandbytes transformers trl
```

Open `fine_tuning_llama2.ipynb` and run cells top to bottom. Requires a Hugging Face token with access to Llama 2 weights.

---

## Results

The fine-tuned model is available on Hugging Face: [`WillBarnard/Llama-2-7b-chat-finetune`](https://huggingface.co/WillBarnard/Llama-2-7b-chat-finetune)

Training was monitored via TensorBoard and inference was validated with a custom text-generation pipeline.
