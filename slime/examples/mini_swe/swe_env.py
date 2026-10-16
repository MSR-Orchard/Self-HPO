"""
Environment helpers for the slime RL rollout: load the agent config, resolve a task's image, and create / stop
the task's mini-swe-agent environment.
"""

import asyncio
import logging
import time
from pathlib import Path

import yaml
from jinja2 import StrictUndefined, Template

logger = logging.getLogger(__name__)

try:
    from minisweagent import Environment
except ImportError as e:
    raise ImportError(
        "mini-swe-agent not found. Please install it first:\n"
        "  pip install mini-swe-agent\n"
        f"Original error: {e}"
    )

from .orchard_docker import OrchardDockerEnvironment, OrchardDockerEnvironmentConfig


# ---- environment helpers ----

def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_docker_image_name(instance: dict) -> str:
    image_name = instance.get("image_url", None)
    if image_name is None:
        iid = instance.get("instance_id", "unknown")
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"mirror.gcr.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    else:
        if image_name.startswith("aweaiteam/") or image_name.startswith("swerebench/"):
            image_name = "mirror.gcr.io/" + image_name
        elif image_name.startswith("docker.io/"):
            image_name = "mirror.gcr.io/" + image_name[len("docker.io/"):]
    return image_name


async def create_environment(
    config: dict,
    instance_id: str | None = None,
    instance: dict | None = None,
    startup_command: str | None = None,
) -> Environment:
    overall_start = time.time()
    env_config = config.copy()
    env_type = env_config.get("environment_class", "orchard_docker")
    if env_type != "orchard_docker":
        raise ValueError(f"Unsupported environment type: {env_type}")

    if instance is not None and "image" not in env_config:
        env_config["image"] = get_docker_image_name(instance)

    if instance is not None and "cwd" not in env_config:
        env_config["cwd"] = instance.get("workdir", "/testbed")

    env = None
    try:
        clean_config = {k: v for k, v in env_config.items() if k != "environment_class"}
        env = OrchardDockerEnvironment(config_class=OrchardDockerEnvironmentConfig, **clean_config)
        logger.info("Created OrchardDockerEnvironment, now env.start_container() for %s...", instance_id)
        await env.start_container()
        logger.info("OrchardDockerEnvironment container started for %s.", instance_id)

        env_elapsed = time.time() - overall_start
        logger.info("Environment (%s) created in %.2fs", env_type, env_elapsed)

        pre_commands = (instance or {}).get("pre_commands", "").replace("\\n", "\n").strip()
        startup_command = startup_command + " && " + pre_commands if startup_command else pre_commands

        if startup_command:
            startup_start = time.time()
            if instance is not None:
                startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
            out = await env.execute(startup_command)
            if out.get("returncode", 0) != 0:
                raise RuntimeError(f"Error executing startup command: {out}")
            startup_elapsed = time.time() - startup_start
            logger.info("Startup command executed in %.2fs", startup_elapsed)

        total_elapsed = time.time() - overall_start
        logger.info("Total environment setup completed in %.2fs", total_elapsed)
        env._created_at = time.time()
        return env
    except BaseException:
        if env is not None:
            logger.warning("create_environment failed; cleaning up partially-created env for %s", instance_id)
            try:
                await stop_environment(env)
            except Exception as cleanup_exc:
                logger.warning("Cleanup of partially-created env failed: %s", cleanup_exc)
        raise


async def stop_environment(env: Environment) -> None:
    if hasattr(env, "_created_at"):
        lifetime = time.time() - env._created_at
        logger.info("Environment lived for %.2fs before stopping", lifetime)
    try:
        await asyncio.wait_for(env.cleanup(), timeout=300)
    except BaseException as e:
        logger.warning("Environment cleanup failed: %s: %s", type(e).__name__, e)
