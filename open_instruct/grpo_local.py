"""Local (CPU/MPS) GRPO training loop for single-machine setups."""

from __future__ import annotations

import asyncio
import dataclasses
import math
import os
import time
from typing import Any, Literal

import numpy as np
import torch
from datasets import Dataset
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, get_scheduler

from open_instruct import data_loader as data_loader_lib
from open_instruct import data_types, grpo_utils, logger_utils, model_utils, utils
from open_instruct.dataset_transformation import (
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    RAW_PROMPT_KEY,
    TOOLS_COLUMN_KEY,
    VERIFIER_SOURCE_KEY,
    get_cached_dataset_tulu,
    validate_dataset_tools,
    visualize_token,
)
from open_instruct.ground_truth_utils import RewardConfig, build_all_verifiers, cleanup_all_llm_judge_clients
from open_instruct.rl_utils import masked_mean, pack_sequences
from open_instruct.data_loader import prepare_collated_data_for_workers
from open_instruct.utils import maybe_use_ai2_hf_entity, maybe_use_ai2_wandb_entity

logger = logger_utils.setup_logger(__name__)


@dataclasses.dataclass
class LocalSamplingConfig:
    temperature: float
    top_p: float
    max_tokens: int
    n: int


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(model_config: model_utils.ModelConfig, device: torch.device) -> torch.dtype:
    if model_config.dtype is not None:
        dtype = getattr(torch, model_config.dtype)
    else:
        # Default to fp32 on MPS for stability during sampling.
        dtype = torch.float32
    if device.type == "mps" and dtype == torch.bfloat16:
        logger.warning("bf16 is not supported on MPS; falling back to float16.")
        dtype = torch.float16
    return dtype


def _maybe_adjust_attn_implementation(model_config: model_utils.ModelConfig, device: torch.device) -> None:
    if device.type != "cuda" and model_config.attn_implementation == "flash_attention_2":
        logger.info("Switching attention implementation to 'sdpa' for non-CUDA backend.")
        model_config.attn_implementation = "sdpa"


def _make_tokenizer(tc, model_config: model_utils.ModelConfig) -> PreTrainedTokenizer:
    tc.tokenizer_revision = model_config.model_revision if tc.tokenizer_revision is None else tc.tokenizer_revision
    tc.tokenizer_name_or_path = (
        model_config.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    tokenizer = tc.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Decoder-only models expect left padding for generation.
    tokenizer.padding_side = "left"
    return tokenizer


def _setup_runtime_variables(
    args: grpo_utils.ExperimentConfig, streaming_config: data_loader_lib.StreamingDataLoaderConfig
) -> grpo_utils.ExperimentConfig:
    args.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    streaming_config.dataset_local_cache_dir = os.path.abspath(streaming_config.dataset_local_cache_dir)
    args.num_learners_per_node = [1]
    args.world_size = 1
    args.sequence_parallel_size = 1
    args.num_training_steps = args.total_episodes // (
        streaming_config.num_unique_prompts_rollout * streaming_config.num_samples_per_prompt_rollout
    )
    args.try_launch_beaker_eval_jobs_on_weka = False
    if args.push_to_hub:
        if args.hf_repo_id is None:
            args.hf_repo_id = "open_instruct_dev"
        if args.hf_entity is None:
            args.hf_entity = maybe_use_ai2_hf_entity()
        if args.hf_entity is None:
            args.hf_entity = HfApi().whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:
            args.hf_repo_revision = args.run_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"
    if args.with_tracking and args.wandb_entity is None:
        args.wandb_entity = maybe_use_ai2_wandb_entity()
    return args


def _setup_experiment_tracking(args: grpo_utils.ExperimentConfig, tc, model_config: model_utils.ModelConfig):
    if not args.with_tracking:
        return None
    import wandb

    all_configs = {}
    all_configs.update(**dataclasses.asdict(args), **dataclasses.asdict(tc), **dataclasses.asdict(model_config))
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        config=all_configs,
        name=args.run_name,
        save_code=True,
        tags=[args.exp_name],
    )
    wandb.define_metric("training_step")
    wandb.define_metric("*", step_metric="training_step")
    return wandb.run.url


def _setup_datasets(
    args: grpo_utils.ExperimentConfig,
    tc,
    tokenizer: PreTrainedTokenizer,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    pass_tools_to_chat_template: bool,
    configured_tool_call_names: list[str] | None = None,
) -> tuple[Dataset, Dataset | None]:
    system_prompt_override = None
    if streaming_config.system_prompt_override_file is not None:
        logger.info(f"Loading system prompt override from {streaming_config.system_prompt_override_file}")
        with open(streaming_config.system_prompt_override_file) as f:
            system_prompt_override = f.read().strip()

    transform_fn_args = [
        {
            "system_prompt_override": system_prompt_override,
            "tool_definitions": [],
            "pass_tools_to_chat_template": pass_tools_to_chat_template,
        },
        {"max_prompt_token_length": streaming_config.max_prompt_token_length},
    ]
    train_dataset = get_cached_dataset_tulu(
        dataset_mixer_list=streaming_config.dataset_mixer_list,
        dataset_mixer_list_splits=streaming_config.dataset_mixer_list_splits,
        tc=tc,
        dataset_transform_fn=streaming_config.dataset_transform_fn,
        transform_fn_args=transform_fn_args,
        dataset_cache_mode=streaming_config.dataset_cache_mode,
        dataset_config_hash=streaming_config.dataset_config_hash,
        hf_entity=args.hf_entity,
        dataset_local_cache_dir=streaming_config.dataset_local_cache_dir,
        dataset_skip_cache=streaming_config.dataset_skip_cache,
        system_prompt_override=system_prompt_override,
    )

    if configured_tool_call_names and TOOLS_COLUMN_KEY in train_dataset.column_names:
        validate_dataset_tools(train_dataset, configured_tool_call_names, "train_dataset")
        logger.info("train_dataset has per-sample tool activation enabled (local backend does not execute tools).")

    train_dataset = train_dataset.shuffle(seed=args.seed)

    if len(streaming_config.dataset_mixer_eval_list) > 0:
        eval_dataset = get_cached_dataset_tulu(
            dataset_mixer_list=streaming_config.dataset_mixer_eval_list,
            dataset_mixer_list_splits=streaming_config.dataset_mixer_eval_list_splits,
            tc=tc,
            dataset_transform_fn=streaming_config.dataset_transform_fn,
            transform_fn_args=transform_fn_args,
            hf_entity=args.hf_entity,
            dataset_cache_mode=streaming_config.dataset_cache_mode,
            dataset_config_hash=streaming_config.dataset_config_eval_hash,
            dataset_local_cache_dir=streaming_config.dataset_local_cache_dir,
            dataset_skip_cache=streaming_config.dataset_skip_cache,
            system_prompt_override=system_prompt_override,
        )
        if configured_tool_call_names and TOOLS_COLUMN_KEY in eval_dataset.column_names:
            validate_dataset_tools(eval_dataset, configured_tool_call_names, "eval_dataset")
        if streaming_config.shuffle_eval_dataset:
            eval_dataset = eval_dataset.shuffle(seed=args.seed)
    else:
        eval_dataset = None

    visualize_token(train_dataset[0][INPUT_IDS_PROMPT_KEY], tokenizer)
    return train_dataset, eval_dataset


def _compute_advantages(
    scores: np.ndarray, num_samples_per_prompt: int, normalization: Literal["standard", "centered"]
) -> np.ndarray:
    scores_per_prompt = scores.reshape(-1, num_samples_per_prompt)
    mean_grouped = scores_per_prompt.mean(axis=-1)
    mean_grouped = np.repeat(mean_grouped, num_samples_per_prompt, axis=0)
    if normalization == "standard":
        std_grouped = scores_per_prompt.std(axis=-1)
        std_grouped = np.repeat(std_grouped, num_samples_per_prompt, axis=0)
        return (scores - mean_grouped) / (std_grouped + 1e-8)
    if normalization == "centered":
        return scores - mean_grouped
    raise ValueError(f"Invalid advantage normalization type: {normalization}")


def _build_request_info(num_responses: int) -> data_types.RequestInfo:
    return data_types.RequestInfo(
        num_calls=[0] * num_responses,
        timeouts=[0] * num_responses,
        tool_errors=[""] * num_responses,
        tool_outputs=[""] * num_responses,
        tool_runtimes=[0.0] * num_responses,
        tool_calleds=[False] * num_responses,
        tool_call_stats=[[] for _ in range(num_responses)],
        excess_tool_calls=[{} for _ in range(num_responses)],
    )


def _generate_responses(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompts: list[list[int]],
    generation_config: LocalSamplingConfig,
    device: torch.device,
) -> tuple[list[list[int]], list[str]]:
    if not prompts:
        return [], []

    prompt_lengths = [len(p) for p in prompts]
    max_prompt_len = max(prompt_lengths)
    pad_token_id = tokenizer.pad_token_id

    if tokenizer.padding_side == "left":
        padded_prompts = [[pad_token_id] * (max_prompt_len - len(p)) + p for p in prompts]
    else:
        padded_prompts = [p + [pad_token_id] * (max_prompt_len - len(p)) for p in prompts]
    input_ids = torch.tensor(padded_prompts, dtype=torch.long, device=device)
    attention_mask = (input_ids != pad_token_id).long()

    n = generation_config.n
    if n > 1:
        input_ids = input_ids.repeat_interleave(n, dim=0)
        attention_mask = attention_mask.repeat_interleave(n, dim=0)
        prompt_lengths = [pl for pl in prompt_lengths for _ in range(n)]

    model.eval()
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            max_new_tokens=generation_config.max_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    model.train()

    responses: list[list[int]] = []
    finish_reasons: list[str] = []
    for seq, prompt_len in zip(outputs, prompt_lengths):
        seq_list = seq.tolist()
        response = seq_list[prompt_len:]
        responses.append(response)
        finish_reasons.append("stop" if tokenizer.eos_token_id in response else "length")
    return responses, finish_reasons


def _compute_response_logprobs(
    model: torch.nn.Module,
    input_ids: list[list[int]],
    prompt_lengths: list[int],
    sequence_lengths: list[int],
    pad_token_id: int,
    temperature: float,
    device: torch.device,
) -> list[list[float]]:
    if not input_ids:
        return []

    max_len = max(len(seq) for seq in input_ids)
    padded = [seq + [pad_token_id] * (max_len - len(seq)) for seq in input_ids]
    input_tensor = torch.tensor(padded, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_tensor, dtype=torch.long)
    for i, seq_len in enumerate(sequence_lengths):
        attention_mask[i, :seq_len] = 1
    position_ids = attention_mask.cumsum(dim=1) - 1
    position_ids = position_ids.clamp(min=0)

    with torch.no_grad():
        logprobs_BT, _ = grpo_utils.forward_for_logprobs(
            model,
            input_tensor,
            attention_mask,
            position_ids,
            pad_token_id,
            temperature,
            return_entropy=False,
        )

    logprobs_list: list[list[float]] = []
    for i, prompt_len in enumerate(prompt_lengths):
        response_len = sequence_lengths[i] - prompt_len
        start = max(prompt_len - 1, 0)
        end = start + response_len
        response_logprobs = logprobs_BT[i, start:end].detach().cpu().tolist()
        logprobs_list.append(response_logprobs)
    return logprobs_list


def _prepare_collated_batch(
    tokenizer: PreTrainedTokenizer,
    prompts: list[list[int]],
    responses: list[list[int]],
    response_logprobs: list[list[float]],
    advantages: np.ndarray,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    dp_world_size: int,
    per_device_train_batch_size: int,
) -> data_types.CollatedBatchData:
    masks = [[1] * len(resp) for resp in responses]
    packed_sequences = pack_sequences(
        queries=prompts,
        responses=responses,
        masks=masks,
        pack_length=streaming_config.pack_length,
        pad_token_id=tokenizer.pad_token_id,
        vllm_logprobs=response_logprobs,
        min_num_batches=dp_world_size,
        mask_tool_use=streaming_config.mask_tool_use,
    )
    lookup_advantages = np.zeros(len(advantages) + 1, dtype=np.float32)
    lookup_advantages[1:] = advantages
    packed_sequences.advantages = [
        torch.tensor(lookup_advantages[packed_mask], dtype=torch.float32)
        for packed_mask in packed_sequences.response_masks
    ]
    collated = prepare_collated_data_for_workers(
        packed_sequences,
        dp_world_size=dp_world_size,
        per_device_train_batch_size=per_device_train_batch_size,
        pad_token_id=tokenizer.pad_token_id,
        pin_memory=False,
    )
    return collated[0]


def _calculate_token_counts(accumulation_steps: int, data_BT: data_types.CollatedBatchData) -> dict[int, float]:
    accumulation_counts: dict[int, float] = {}
    local_counts = [mask[:, 1:].sum().float() for mask in data_BT.response_masks]
    if not local_counts:
        return accumulation_counts
    for i, count in enumerate(local_counts):
        group_idx = i // accumulation_steps
        key = int(group_idx * accumulation_steps)
        accumulation_counts[key] = accumulation_counts.get(key, 0.0) + count.item()
    return accumulation_counts


def _train_step_local(
    model: torch.nn.Module,
    ref_policy: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scheduler,
    data_BT: data_types.CollatedBatchData,
    args: grpo_utils.ExperimentConfig,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, float]:
    for f in dataclasses.fields(data_BT):
        tensors = getattr(data_BT, f.name)
        for i in range(len(tensors)):
            tensors[i] = tensors[i].to(device)
    data_BT.response_masks = [mask.bool() for mask in data_BT.response_masks]

    num_samples = len(data_BT)
    accumulation_steps = max(math.ceil(num_samples / args.num_mini_batches - 0.5), 1)
    leftover = num_samples % accumulation_steps
    if leftover > 0:
        data_BT = data_BT[:-leftover]
        logger.warning(f"{leftover} samples dropped due to num_mini_batches={args.num_mini_batches}")

    ref_logprobs_BT: list[torch.Tensor] = []
    if args.load_ref_policy and ref_policy is not None:
        ref_policy.eval()
        ref_logprobs_BT = grpo_utils.compute_logprobs(
            ref_policy, data_BT, pad_token_id, streaming_config.temperature, use_grad=False
        )

    old_logprobs_BT: list[torch.Tensor | None] = [None for _ in range(len(data_BT.query_responses))]
    if args.use_vllm_logprobs:
        for i in range(len(data_BT.query_responses)):
            vllm_old_logprob_BT = data_BT.vllm_logprobs[i][:, 1:]
            vllm_old_logprob_BT = torch.masked_fill(
                vllm_old_logprob_BT, ~data_BT.response_masks[i][:, 1:], utils.INVALID_LOGPROB
            )
            old_logprobs_BT[i] = torch.nan_to_num(vllm_old_logprob_BT, nan=utils.INVALID_LOGPROB)
    else:
        local_old_logprobs_BT = grpo_utils.compute_logprobs(
            model, data_BT, pad_token_id, streaming_config.temperature, use_grad=False
        )
        for i in range(len(data_BT.query_responses)):
            old_logprobs_BT[i] = local_old_logprobs_BT[i]

    num_samples = len(data_BT.query_responses)
    token_counts_per_sample = torch.stack([mask[:, 1:].sum().float() for mask in data_BT.response_masks])
    total_valid_tokens = token_counts_per_sample.sum().item()

    loss_stats = {
        "loss": torch.zeros(num_samples, device=device),
        "pg_loss": torch.zeros(num_samples, device=device),
        "kl_loss": torch.zeros(num_samples, device=device),
    }

    local_step = 0
    for epoch_idx in range(args.num_epochs):
        if args.loss_denominator == "token":
            accumulation_token_counts = _calculate_token_counts(accumulation_steps, data_BT)
        else:
            accumulation_token_counts = {
                int(group_idx * accumulation_steps): float(args.loss_denominator)
                for group_idx in range((len(data_BT.query_responses) // accumulation_steps) + 1)
            }

        for i in range(num_samples):
            response_mask_BT = data_BT.response_masks[i][:, 1:]
            batch_start = (i // accumulation_steps) * accumulation_steps
            loss_denominator = accumulation_token_counts[batch_start]

            local_logprobs_BT, _ = grpo_utils.forward_for_logprobs(
                model,
                data_BT.query_responses[i],
                data_BT.attention_masks[i],
                data_BT.position_ids[i],
                pad_token_id,
                streaming_config.temperature,
                return_entropy=False,
            )
            local_logprobs_BT = torch.masked_fill(local_logprobs_BT, ~response_mask_BT, utils.INVALID_LOGPROB)

            old_logprob_BT = old_logprobs_BT[i]
            assert old_logprob_BT is not None

            logprobs_diff_BT = local_logprobs_BT - old_logprob_BT
            ratio_BT = torch.exp(logprobs_diff_BT)

            pg_losses_BT, pg_losses2_BT, pg_loss_max_BT, kl_BT = grpo_utils.compute_grpo_loss(
                new_logprobs=local_logprobs_BT,
                ratio=ratio_BT,
                advantages=data_BT.advantages[i][:, 1:],
                ref_logprobs=ref_logprobs_BT[i] if args.load_ref_policy else None,
                config=args,
            )

            per_token_loss_BT = pg_loss_max_BT + args.beta * kl_BT
            loss = masked_mean(per_token_loss_BT, response_mask_BT, None, loss_denominator)

            loss.backward()
            if (local_step + 1) % accumulation_steps == 0:
                if args.max_grad_norm and args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
            local_step += 1

            with torch.no_grad():
                loss_stats["loss"][i] = loss.detach()
                loss_stats["pg_loss"][i] = masked_mean(pg_loss_max_BT, response_mask_BT)
                loss_stats["kl_loss"][i] = masked_mean(kl_BT, response_mask_BT) * args.beta

    metrics = {
        "loss/total_avg": loss_stats["loss"].mean().item(),
        "loss/policy_avg": loss_stats["pg_loss"].mean().item(),
        "loss/kl_avg": loss_stats["kl_loss"].mean().item(),
        "tokens/total": total_valid_tokens,
    }
    return metrics


def _save_local_model(
    model: torch.nn.Module, tokenizer: PreTrainedTokenizer, output_dir: str, step: int | None = None
) -> None:
    save_dir = output_dir if step is None else os.path.join(output_dir, f"step_{step}")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=False)
    tokenizer.save_pretrained(save_dir)


def main_local(
    args: grpo_utils.ExperimentConfig,
    tc,
    model_config: model_utils.ModelConfig,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    vllm_config: data_loader_lib.VLLMConfig,
    tools_config,
) -> None:
    if tools_config.enabled:
        logger.warning("Local backend does not execute tools; tool calls will be treated as plain text.")

    device = _select_device()
    _maybe_adjust_attn_implementation(model_config, device)
    dtype = _resolve_dtype(model_config, device)

    args = _setup_runtime_variables(args, streaming_config)
    args.temperature = streaming_config.temperature
    os.makedirs(args.output_dir, exist_ok=True)

    wandb_url = _setup_experiment_tracking(args, tc, model_config)

    tokenizer = _make_tokenizer(tc, model_config)

    train_dataset, eval_dataset = _setup_datasets(
        args,
        tc,
        tokenizer,
        streaming_config,
        pass_tools_to_chat_template=tools_config.pass_tools_to_chat_template,
        configured_tool_call_names=tools_config.tool_call_names if tools_config.enabled else None,
    )

    if args.cache_dataset_only:
        return

    logger.info(f"Using device={device}, dtype={dtype}, attn={model_config.attn_implementation}")
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        dtype=dtype,
        attn_implementation=model_config.attn_implementation,
        use_cache=True,
    ).to(device)

    if model_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ref_policy = None
    if args.load_ref_policy:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            dtype=dtype,
            attn_implementation=model_config.attn_implementation,
            use_cache=False,
        ).to(device)
        ref_policy.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    num_scheduler_steps = args.num_training_steps * args.num_epochs * args.num_mini_batches
    warm_up_steps = args.warm_up_steps
    if args.warmup_ratio > 0.0:
        warm_up_steps = int(num_scheduler_steps * args.warmup_ratio)
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warm_up_steps,
        num_training_steps=num_scheduler_steps,
    )

    reward_config = RewardConfig(
        apply_r1_style_format_reward=streaming_config.apply_r1_style_format_reward,
        r1_style_format_reward=streaming_config.r1_style_format_reward,
        apply_verifiable_reward=streaming_config.apply_verifiable_reward,
        verification_reward=streaming_config.verification_reward,
        non_stop_penalty=streaming_config.non_stop_penalty,
        non_stop_penalty_value=streaming_config.non_stop_penalty_value,
        only_reward_good_outputs=tools_config.only_reward_good_outputs,
        additive_format_reward=streaming_config.additive_format_reward,
        verifier_functions=build_all_verifiers(args, streaming_config),
    )
    reward_fn = reward_config.build()
    if not args.use_vllm_logprobs:
        logger.info("Local backend will use generator logprobs as old logprobs.")
        args.use_vllm_logprobs = True

    train_iter = iter(train_dataset)
    num_steps = args.num_training_steps
    n = streaming_config.num_samples_per_prompt_rollout
    generation_config = LocalSamplingConfig(
        temperature=streaming_config.temperature,
        top_p=vllm_config.vllm_top_p,
        max_tokens=streaming_config.response_length,
        n=n,
    )
    if streaming_config.stop_strings:
        logger.warning("Local backend ignores stop_strings; only eos_token_id stopping is used.")

    for step in range(1, num_steps + 1):
        prompts_batch = []
        prompt_ids = []
        ground_truths = []
        verifier_sources = []
        raw_prompts = []
        for _ in range(streaming_config.num_unique_prompts_rollout):
            try:
                sample = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataset)
                sample = next(train_iter)
            prompt_tokens = sample[INPUT_IDS_PROMPT_KEY]
            if torch.is_tensor(prompt_tokens):
                prompt_tokens = prompt_tokens.tolist()
            prompts_batch.append(prompt_tokens)
            prompt_ids.append(sample.get("index", None))
            ground_truths.append(sample[GROUND_TRUTHS_KEY])
            verifier_sources.append(sample[VERIFIER_SOURCE_KEY])
            raw_prompts.append(sample.get(RAW_PROMPT_KEY, ""))

        responses, finish_reasons = _generate_responses(
            model, tokenizer, prompts_batch, generation_config=generation_config, device=device
        )
        prompt_lengths = [len(p) for p in prompts_batch for _ in range(n)]
        flattened_prompts = [p for p in prompts_batch for _ in range(n)]
        full_sequences = [p + r for p, r in zip(flattened_prompts, responses)]
        sequence_lengths = [len(seq) for seq in full_sequences]
        response_logprobs = _compute_response_logprobs(
            model,
            full_sequences,
            prompt_lengths,
            sequence_lengths,
            tokenizer.pad_token_id,
            streaming_config.temperature,
            device,
        )

        decoded_responses = tokenizer.batch_decode(responses, skip_special_tokens=True)
        k_ground_truths = [gt for gt in ground_truths for _ in range(n)]
        k_datasets = [ds for ds in verifier_sources for _ in range(n)]
        k_raw_queries = [rq for rq in raw_prompts for _ in range(n)]

        request_info = _build_request_info(len(responses))
        scores, reward_metrics = asyncio.run(
            reward_fn(
                responses,
                decoded_responses,
                k_ground_truths,
                k_datasets,
                finish_reasons,
                request_info,
                queries=k_raw_queries,
            )
        )
        scores_np = np.array(scores, dtype=np.float32)
        advantages = _compute_advantages(
            scores_np, streaming_config.num_samples_per_prompt_rollout, streaming_config.advantage_normalization_type
        )

        collated = _prepare_collated_batch(
            tokenizer,
            flattened_prompts,
            responses,
            response_logprobs,
            advantages,
            streaming_config,
            dp_world_size=1,
            per_device_train_batch_size=args.per_device_train_batch_size,
        )

        metrics = _train_step_local(
            model,
            ref_policy,
            optimizer,
            scheduler,
            collated,
            args,
            streaming_config,
            tokenizer.pad_token_id,
            device,
        )
        metrics["training_step"] = step
        metrics["reward/mean"] = float(scores_np.mean()) if len(scores_np) else 0.0
        for k, v in reward_metrics.items():
            metrics[k] = v

        if args.save_freq > 0 and step % args.save_freq == 0:
            logger.info(f"Saving checkpoint at step {step}")
            _save_local_model(model, tokenizer, args.output_dir, step=step)

        if args.with_tracking:
            import wandb

            wandb.log(metrics)

        if step % 10 == 0 or step == 1:
            logger.info(
                f"[local] step={step} loss={metrics['loss/total_avg']:.4f} reward_mean={metrics['reward/mean']:.3f}"
            )

    _save_local_model(model, tokenizer, args.output_dir, step=None)
    if args.push_to_hub and args.hf_repo_id:
        model_utils.push_folder_to_hub(args.output_dir, args.hf_repo_id, args.hf_repo_revision)

    asyncio.run(cleanup_all_llm_judge_clients())
    logger.info("finished local training")
