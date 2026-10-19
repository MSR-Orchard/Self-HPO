"""
Custom generate function for SWE-bench tasks with a tool-call based mini-swe-agent.
"""

import asyncio
import json
import logging
import os
import random
from datetime import datetime

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .swe_agent import (
    TOOLS,
    EnvironmentUnavailable,
    FormatError,
    LimitsExceeded,
    SWEAgent,
    TimeExceeded,
)
from .swe_env import create_environment, load_config, stop_environment

DEFAULT_MAX_ALL_TOKENS = 65536 - 4096*2
logger = logging.getLogger(__name__)

# ---- timeouts (seconds) ----
TIMEOUT_CREATE_ENV = int(os.environ["SWE_TIMEOUT_CREATE_ENV"])
TIMEOUT_LLM_INFERENCE = int(os.environ["SWE_TIMEOUT_LLM_INFERENCE"])
TIMEOUT_GET_OBSERVATION = int(os.environ["SWE_TIMEOUT_GET_OBSERVATION"])
TIMEOUT_STOP_ENV = int(os.environ["SWE_TIMEOUT_STOP_ENV"])
RETURN_LOGPROB = True

# ---- retries ----
MAX_GENERATE_ENV_RETRIES = int(os.environ["SWE_MAX_ENV_RETRIES"])
GENERATE_ENV_RETRY_WAIT = int(os.environ["SWE_ENV_RETRY_WAIT"])

MAX_CREATE_ENV_RETRIES = int(os.environ["SWE_MAX_CREATE_ENV_RETRIES"])
CREATE_ENV_RETRY_WAIT = int(os.environ["SWE_CREATE_ENV_RETRY_WAIT"])

ENV_CREATE_JITTER_MAX = int(os.environ["SWE_ENV_CREATE_JITTER_MAX"])

_run_timestamp = None


def get_token_delta(
    tokenizer,
    messages: list[dict],
    previous_len: int = -1,
    tools: list[dict] | None = None,
) -> tuple[list[int], list[int]]:
    token_ids = []
    loss_mask = []
    previous_len = previous_len if previous_len >= 0 else len(messages) + previous_len
    current_len = len(messages)
    if 0 < previous_len < current_len and all(m["role"] != "assistant" for m in messages[previous_len:]):
        prev = tokenizer.apply_chat_template(
            messages[:previous_len], tokenize=False, add_generation_prompt=False, tools=tools
        )
        curr = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools)
        if not curr.startswith(prev):
            marker = "\x00token-delta\x00"
            ref = tokenizer.apply_chat_template(
                messages[:previous_len] + [{"role": "user", "content": marker}],
                tokenize=False, add_generation_prompt=False, tools=tools,
            )
            cut = ref.rfind("<|im_start|>user\n" + marker)
            if cut <= 0 or not curr.startswith(ref[:cut]):
                raise ValueError("cannot locate the new messages in the rendered conversation")
            new_tokens = tokenizer.encode(curr[cut:])
            return new_tokens, [0] * len(new_tokens)
    for l in range(previous_len, current_len):
        if l == 0:
            prev = ""
        else:
            if messages[:l][-1]["role"] == "assistant":
                prev = tokenizer.apply_chat_template(
                    messages[:l], tokenize=False, add_generation_prompt=False, tools=tools
                )
            else:
                prev = tokenizer.apply_chat_template(
                    messages[:l], tokenize=False, add_generation_prompt=True, tools=tools
                )
        if messages[:l+1][-1]["role"] == "assistant":
            curr = tokenizer.apply_chat_template(
                messages[:l+1], tokenize=False, add_generation_prompt=False, tools=tools
            )
            new_tokens = tokenizer.encode(curr[len(prev):])
            token_ids += new_tokens
            loss_mask += [1] * len(new_tokens)
        else:
            curr = tokenizer.apply_chat_template(
                messages[:l+1], tokenize=False, add_generation_prompt=True, tools=tools
            )
            new_tokens = tokenizer.encode(curr[len(prev):])
            token_ids += new_tokens
            loss_mask += [0] * len(new_tokens)
    return token_ids, loss_mask

async def generate(args, sample: Sample, sampling_params) -> Sample:
    resource_level = 1
    for gen_attempt in range(1, MAX_GENERATE_ENV_RETRIES + 1):
        try:
            return await _generate_once(args, sample, sampling_params, resource_level)
        except EnvironmentUnavailable as e:
            logger.error(
                f"[SWE_GENERATE] EnvironmentUnavailable on attempt "
                f"{gen_attempt}/{MAX_GENERATE_ENV_RETRIES}: {e}"
            )
            resource_level += 1
            if gen_attempt < MAX_GENERATE_ENV_RETRIES:
                logger.info(
                    f"[SWE_GENERATE] Retrying generate in {GENERATE_ENV_RETRY_WAIT}s..."
                )
                await asyncio.sleep(GENERATE_ENV_RETRY_WAIT)
            else:
                logger.error(
                    f"[SWE_GENERATE] All {MAX_GENERATE_ENV_RETRIES} retry attempts "
                    f"exhausted, aborting sample"
                )
                sample.status = Sample.Status.ABORTED
                return sample
        except Exception as e:
            logger.error(
                f"[SWE_GENERATE] Unexpected error on attempt "
                f"{gen_attempt}/{MAX_GENERATE_ENV_RETRIES}: {e}"
            )
            if gen_attempt < MAX_GENERATE_ENV_RETRIES:
                logger.info(
                    f"[SWE_GENERATE] Retrying generate in {GENERATE_ENV_RETRY_WAIT}s..."
                )
                await asyncio.sleep(GENERATE_ENV_RETRY_WAIT)
            else:
                logger.error(
                    f"[SWE_GENERATE] All {MAX_GENERATE_ENV_RETRIES} retry attempts "
                    f"exhausted, aborting sample"
                )
                sample.status = Sample.Status.ABORTED
                return sample


async def _generate_once(args, sample: Sample, sampling_params, resource_level: int = 1) -> Sample:
    import time
    generate_start_time = time.time()

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    config_path = getattr(args, "swe_config_path", None)
    if config_path is None:
        config_path = os.environ["SWE_CONFIG_PATH"]
    config = load_config(config_path)
    agent_config = config.get("agent", {})

    multiply = sampling_params.get("max_new_tokens") // 4096
    if multiply < 1:
        multiply = 1

    MAX_LLM_ATTEMPTS = 2
    max_all_tokens = agent_config.get("max_all_tokens", DEFAULT_MAX_ALL_TOKENS) * multiply
    llm_inference_timeout = TIMEOUT_LLM_INFERENCE * multiply
    max_llm_attempts = MAX_LLM_ATTEMPTS * multiply
    if agent_config.get("time_limit", 0) > 0:
        agent_config["time_limit"] = agent_config["time_limit"] * multiply
        logger.info(f"[SWE_GENERATE] Scaled time_limit to {agent_config['time_limit']:.0f}s (multiply={multiply})")
    logger.info(f"Set max_all_tokens={max_all_tokens} based on config and sampling_params max_new_tokens={sampling_params.get('max_new_tokens')}")

    env_config = config.get("environment", {})

    env_config["cpu"] = str(resource_level * 2)
    env_config["memory"] = f"{resource_level * 8}Gi"
    logger.info(f"[SWE_GENERATE] Environment resources for resource_level {resource_level}: cpu={env_config['cpu']}, memory={env_config['memory']}")

    instance_id = sample.metadata.get("instance_id", f"task_{sample.index}")
    logger.info(f"[SWE_GENERATE] Starting generate for instance_id={instance_id}, sample_index={sample.index}")
    problem_statement = (
        sample.prompt if isinstance(sample.prompt, str) else sample.prompt[0]["content"]
    )

    instance = {
        "instance_id": instance_id,
        **sample.metadata,
    }

    startup_command = config.get("run", {}).get("env_startup_command")

    logger.info(f"[SWE_GENERATE] start create_environment for instance_id={instance_id}:{sample.index}")
    env_create_start_time = time.time()
    env = None

    if ENV_CREATE_JITTER_MAX > 0:
        jitter = random.uniform(0, ENV_CREATE_JITTER_MAX)
        logger.info(f"[SWE_GENERATE] Sleeping {jitter:.1f}s (jitter) before creating environment for instance_id={instance_id}:{sample.index}")
        await asyncio.sleep(jitter)

    for attempt in range(1, MAX_CREATE_ENV_RETRIES + 1):
        logger.info(f"[SWE_GENERATE] Attempt {attempt}/{MAX_CREATE_ENV_RETRIES} to create environment for instance_id={instance_id}:{sample.index}")
        try:
            env = await asyncio.wait_for(
                create_environment(
                    env_config,
                    instance_id=instance_id + ":" + str(sample.index),
                    instance=instance,
                    startup_command=startup_command,
                ),
                timeout=TIMEOUT_CREATE_ENV,
            )
            break
        except asyncio.TimeoutError:
            logger.error(
                f"[SWE_GENERATE] TIMEOUT creating environment for instance_id={instance_id}:{sample.index} "
                f"after {TIMEOUT_CREATE_ENV}s (attempt {attempt}/{MAX_CREATE_ENV_RETRIES})"
            )
            if attempt < MAX_CREATE_ENV_RETRIES:
                logger.info(f"[SWE_GENERATE] Retrying in {CREATE_ENV_RETRY_WAIT}s for instance_id={instance_id}:{sample.index}")
                await asyncio.sleep(CREATE_ENV_RETRY_WAIT)
            else:
                logger.error(f"[SWE_GENERATE] All {MAX_CREATE_ENV_RETRIES} attempts exhausted for instance_id={instance_id}:{sample.index}")
                sample.status = Sample.Status.ABORTED
                return sample
        except Exception as e:
            logger.error(
                f"[SWE_GENERATE] Failed to create environment for instance_id={instance_id}:{sample.index}: {e} "
                f"(attempt {attempt}/{MAX_CREATE_ENV_RETRIES})"
            )
            if attempt < MAX_CREATE_ENV_RETRIES:
                logger.info(f"[SWE_GENERATE] Retrying in {CREATE_ENV_RETRY_WAIT}s for instance_id={instance_id}:{sample.index}")
                await asyncio.sleep(CREATE_ENV_RETRY_WAIT)
            else:
                logger.error(f"[SWE_GENERATE] All {MAX_CREATE_ENV_RETRIES} attempts exhausted for instance_id={instance_id}:{sample.index}")
                sample.status = Sample.Status.ABORTED
                return sample

    assert env is not None, "Environment creation failed without raising an exception"

    env_creation_time = time.time() - env_create_start_time
    logger.info(f"[SWE_GENERATE] complete create_environment for instance_id={instance_id}:{sample.index}, env_creation_time={env_creation_time:.2f}s, elapsed={time.time() - generate_start_time:.2f}s")
    logger.info(f"[SWE_GENERATE] start agent loop for instance_id={instance_id}:{sample.index}")
    try:
        # ---- setup agent ----
        agent = SWEAgent(env=env, **agent_config)
        workdir = instance.get("workdir", "/testbed")
        agent.setup(task=problem_statement, workdir=workdir)
        logger.info(f"[SWE_GENERATE] complete SWEAgent setup for instance_id={instance_id}:{sample.index}, elapsed={time.time() - generate_start_time:.2f}s")

        prompt_token_ids = []
        init_prompt_text = ""

        response = ""
        response_token_ids = []
        loss_mask = []
        rollout_log_probs = [] if RETURN_LOGPROB else None

        prev_state_response = ""
        prev_state_response_token_ids = []
        prev_state_loss_mask = []
        prev_state_rollout_log_probs = [] if RETURN_LOGPROB else None

        exit_status = "Unknown"
        final_output = ""
        sample.status = Sample.Status.PENDING

        max_steps = agent.config.step_limit if agent.config.step_limit else 500

        # ---- main interaction loop ----
        for step in range(max_steps):
            # ---- check limits ----
            try:
                agent.check_limits()
            except TimeExceeded:
                exit_status = "TimeExceeded"
                break
            except LimitsExceeded:
                exit_status = "LimitsExceeded"
                break

            assert agent.messages[-1]["role"] in ("user", "tool"), \
                f"Error: Last message must be from user or tool, got {agent.messages[-1]['role']}"

            # ---- build prompt ----
            current_prompt = state.tokenizer.apply_chat_template(
                agent.messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=TOOLS,
            )

            if not prompt_token_ids:
                prompt_token_ids = state.tokenizer(current_prompt, add_special_tokens=False)["input_ids"]
            if not init_prompt_text:
                init_prompt_text = current_prompt

            # ---- generate action ----
            payload = {
                "text": current_prompt,
                "sampling_params": sampling_params,
                "return_logprob": RETURN_LOGPROB,
            }

            llm_start_time = time.time()
            output = None
            for llm_attempt in range(1, max_llm_attempts + 1):
                try:
                    output = await asyncio.wait_for(
                        post(url, payload),
                        timeout=llm_inference_timeout,
                    )
                    break
                except asyncio.TimeoutError:
                    logger.error(
                        f"[SWE_GENERATE] TIMEOUT waiting for LLM inference for "
                        f"instance_id={instance_id}:{sample.index}, step={step} after {llm_inference_timeout}s "
                        f"(attempt {llm_attempt}/{max_llm_attempts})"
                    )
                    if llm_attempt >= max_llm_attempts:
                        sample.status = Sample.Status.ABORTED
                        return sample
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error(
                        f"[SWE_GENERATE] Error during LLM inference for instance_id={instance_id}:{sample.index}, step={step}: {e} "
                        f"(attempt {llm_attempt}/{max_llm_attempts})"
                    )
                    if llm_attempt >= max_llm_attempts:
                        sample.status = Sample.Status.ABORTED
                        return sample
                    await asyncio.sleep(5)
            llm_inference_time = time.time() - llm_start_time
            logger.info(f"[SWE_GENERATE] LLM inference for instance_id={instance_id}:{sample.index}, step={step}, time={llm_inference_time:.2f}s")

            if output["meta_info"]["finish_reason"]["type"] == "abort":
                sample.status = Sample.Status.ABORTED
                return sample

            cur_response = output["text"]

            if RETURN_LOGPROB:
                if "output_token_logprobs" not in output["meta_info"]:
                    raise RuntimeError(
                        "output_token_logprobs not found in output meta_info. "
                        "Make sure 'return_logprob': True is set in the payload."
                    )
                cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
                cur_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            else:
                cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]

            if cur_response.endswith(state.tokenizer.eos_token):
                cur_response = cur_response[: -len(state.tokenizer.eos_token)]
            if "</think>" in cur_response and not cur_response.startswith("<think>"):
                if "\n</think>" in cur_response:
                    cur_response = "<think>\n" + cur_response
                else:
                    cur_response = "<think>" + cur_response

            cur_message = {
                "role": "assistant",
                "content": cur_response,
            }

            response += cur_response
            response_token_ids += cur_response_token_ids
            loss_mask += [1] * len(cur_response_token_ids)
            agent.n_calls += 1

            if RETURN_LOGPROB:
                rollout_log_probs += cur_response_log_probs

            agent.add_messages(cur_message)
            previous_message_len = len(agent.messages)

            if len(prev_state_response_token_ids) == 0:
                prev_state_response_token_ids = response_token_ids.copy()
                prev_state_loss_mask = loss_mask.copy()
                if RETURN_LOGPROB:
                    prev_state_rollout_log_probs = rollout_log_probs.copy()
                prev_state_response = response

            if output["meta_info"]["finish_reason"]["type"] == "length":
                exit_status = "LimitsExceeded"
                break

            # ---- parse response ----
            try:
                logger.info(f"[SWE_GENERATE] start parse_response for instance_id={instance_id}:{sample.index}, step={step}")
                parse_response_result = agent.parse_response(cur_response)
                agent.messages[-1]["extra"] = parse_response_result.get("extra", {})
                # ---- execute actions ----
                actions = parse_response_result.get("extra", {}).get("actions", [])
                outputs, submission_info = await agent.execute_actions(
                    actions, timeout=TIMEOUT_GET_OBSERVATION
                )
                if submission_info is not None:
                    final_output = submission_info["submission"]
                    exit_status = "Submitted"
                    break
                obs_messages = agent.format_observation_messages(agent.messages[-1], outputs)
                for obs_msg in obs_messages:
                    agent.add_messages(obs_msg)

            except FormatError as e:
                logger.error(f"[SWE_GENERATE] FormatError for instance_id={instance_id}:{sample.index}, step={step}: {e}")
                agent.add_messages(*e.messages)

            except EnvironmentUnavailable:
                raise

            except Exception as e:
                logger.error(f"[SWE_GENERATE] Exception during action execution for instance_id={instance_id}:{sample.index}, step={step}: {e}")
                error_message = f"Error: unexpected failure during action execution: {e}"
                agent.add_messages({"role": "tool", "content": error_message})


            obs_token_ids, obs_mask = get_token_delta(
                state.tokenizer, agent.messages, previous_message_len, tools=TOOLS
            )

            response_token_ids += obs_token_ids
            loss_mask += obs_mask
            if RETURN_LOGPROB:
                rollout_log_probs += [0.0] * len(obs_token_ids)

            if RETURN_LOGPROB:
                assert len(response_token_ids) == len(
                    rollout_log_probs
                ), f"Token/logp length mismatch: {len(response_token_ids)} tokens vs {len(rollout_log_probs)} logps"

            if len(prompt_token_ids) + len(response_token_ids) >= max_all_tokens:
                exit_status = "LimitsExceeded"
                break

            prev_state_response_token_ids = response_token_ids.copy()
            prev_state_loss_mask = loss_mask.copy()
            if RETURN_LOGPROB:
                prev_state_rollout_log_probs = rollout_log_probs.copy()
            prev_state_response = response

        if step == max_steps - 1 and exit_status == "Unknown":
            exit_status = "LimitsExceeded"

        if exit_status in ("LimitsExceeded", "TimeExceeded") and len(prompt_token_ids) + len(response_token_ids) >= max_all_tokens:
            response_token_ids = prev_state_response_token_ids
            loss_mask = prev_state_loss_mask
            if RETURN_LOGPROB:
                rollout_log_probs = prev_state_rollout_log_probs
            response = prev_state_response

        sample.tokens = prompt_token_ids + response_token_ids
        sample.response_length = len(response_token_ids)
        sample.response = response
        sample.loss_mask = loss_mask
        sample.prompt = init_prompt_text

        if RETURN_LOGPROB:
            sample.rollout_log_probs = rollout_log_probs

        if exit_status == "Submitted":
            sample.status = Sample.Status.COMPLETED
        elif exit_status in ("LimitsExceeded", "TimeExceeded"):
            sample.status = Sample.Status.TRUNCATED
        elif exit_status in ("Timeout_LLM", "Error_LLM"):
            sample.status = Sample.Status.ABORTED
        else:
            match output["meta_info"]["finish_reason"]["type"]:
                case "length":
                    sample.status = Sample.Status.TRUNCATED
                case "abort":
                    sample.status = Sample.Status.ABORTED
                case "stop":
                    sample.status = Sample.Status.COMPLETED

        if sample.status == Sample.Status.PENDING:
            sample.status = Sample.Status.TRUNCATED

        logger.info(f"[SWE_GENERATE] complete agent loop for instance_id={instance_id}:{sample.index}")

        global _run_timestamp
        if _run_timestamp is None:
            _run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        WANDB_RANDOM_SUFFIX = os.environ.get("WANDB_RANDOM_SUFFIX", "")
        trajectory_dir = os.path.join(
            os.environ.get("SWE_TRAJECTORY_DIR", "trajectories"),
            _run_timestamp + (f"_{WANDB_RANDOM_SUFFIX}" if WANDB_RANDOM_SUFFIX else ""),
        )
        os.makedirs(trajectory_dir, exist_ok=True)
        trajectory_path = os.path.join(trajectory_dir, f"{instance_id}_{sample.index}.json")
        group_info_path = os.path.join(trajectory_dir, f"{instance_id}_group_info.jsonl")

        sample.metadata.update({
            "trajectory_path": trajectory_path,
            "group_info_path": group_info_path,
            "instance_id": instance_id,
            "trajectory": agent.serialize(),
            "n_steps": agent.n_calls,
            "exit_status": exit_status,
            "final_output": final_output,
            "env_creation_time": env_creation_time,
        })

        trajectory_data = {
            "instance_id": instance_id,
            "instance": instance,
            "messages": agent.messages,
            "model_patch": final_output,
            "n_steps": agent.n_calls,
            "exit_status": exit_status,
            "all_tokens_length": len(prompt_token_ids) + len(response_token_ids),
            "prompt_token_ids": prompt_token_ids,
            "response_token_ids": response_token_ids,
            "loss_mask": loss_mask,
            "env_creation_time": env_creation_time,
        }
        with open(trajectory_path, "w") as f:
            json.dump(trajectory_data, f, indent=2, default=str)
        logger.info("Saved trajectory to %s", trajectory_path)

        if sample.rollout_log_probs is not None:
            assert len(response_token_ids) == len(sample.rollout_log_probs), (
                f"{instance_id}_{sample.index} token/logp mismatch: {len(response_token_ids)} vs {len(sample.rollout_log_probs)}"
            )
        assert len(loss_mask) == len(response_token_ids), (
            f"{instance_id}_{sample.index} loss mask/token mismatch: {len(loss_mask)} vs {len(response_token_ids)}"
        )

        logger.info(f"[SWE_GENERATE] Completed generate for instance_id={instance_id}, sample_index={sample.index}, exit_status={exit_status}, total_time={time.time() - generate_start_time:.2f}s")
        return sample
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, stopping remote environment...")
        raise
    except Exception as e:
        logger.error(f"[SWE_GENERATE] Exception in generate for instance_id={instance_id}, sample_index={sample.index}: {e}")
        raise
    finally:
        logger.info(f"[SWE_GENERATE] Stopping environment for instance_id={instance_id}, sample_index={sample.index}")
        try:
            await asyncio.wait_for(stop_environment(env), timeout=TIMEOUT_STOP_ENV)
        except BaseException as e:
            logger.error(f"[SWE_GENERATE] Error stopping environment for instance_id={instance_id}: {type(e).__name__}: {e} – environment may be leaked")
        logger.info(f"[SWE_GENERATE] Environment stopped for instance_id={instance_id}, sample_index={sample.index}")
