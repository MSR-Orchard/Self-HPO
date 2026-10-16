"""Orchard sandbox environment used by the slime rollout and reward modules."""
import asyncio
import logging
from dataclasses import asdict, dataclass, field
from typing import Any
import random
from rich.markup import escape as rich_escape


@dataclass
class OrchardDockerEnvironmentConfig:
    image: str
    cwd: str = "/"
    env: dict[str, str] = field(default_factory=dict)
    forward_env: list[str] = field(default_factory=list)
    exe_timeout: int = 900
    cpu: str = "2"
    memory: str = "8Gi"
    container_ready_timeout: int = 1200
    cleanup_timeout: int = 300
    heartbeat_interval: int = 90


class OrchardDockerEnvironment:
    def __init__(
        self,
        *,
        config_class: type = OrchardDockerEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.container = None
        self._sandbox_id = None
        self.config = config_class(**kwargs)
        self.logger.info(f"Using configuration: {self.config}")

        from orchard_env import AsyncSandboxClient

        self.client = AsyncSandboxClient()

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    async def start_container(self):
        self.logger.info(f"Starting container with image {self.config.image}")

        self.logger.info(f"Creating sandbox for {self.config.image}")
        self.container = await self.client.create_sandbox(
                            image=self.config.image,
                            cpu=self.config.cpu,
                            memory=self.config.memory,
                            timeout=self.config.container_ready_timeout,
                            block_network=False,
                        )
        self._sandbox_id = getattr(self.container, "sandbox_id", None)
        self.logger.info(f"Sandbox created for {self.config.image}")

        if self.config.heartbeat_interval > 0:
            jittered_interval = self.config.heartbeat_interval + random.randint(-30, 30)
            jittered_interval = max(30, jittered_interval)
            self.logger.info(f"Starting heartbeat task with interval {jittered_interval} seconds (base: {self.config.heartbeat_interval})")
            self.container.start_heartbeat(interval=jittered_interval)


        bashrc_command = f"echo 'cd {self.config.cwd}' >> /root/.bashrc"
        try:
            result = await asyncio.wait_for(
                self.container.exec(
                    bashrc_command, timeout=self.config.exe_timeout, cwd=self.config.cwd,
                    login_shell=True, env=None,
                ),
                timeout=self.config.exe_timeout + 60,
            )
        except asyncio.TimeoutError:
            self.logger.error("Timed out executing bashrc setup command in new container")
            raise
        except Exception as e:
            self.logger.error(f"Error executing bashrc setup command in new container: {e}")
            raise
        if result.stderr:
            self.logger.info(f"Executed bashrc command with output: {result.stdout}, error: {result.stderr}")

    async def execute(self, command: str) -> dict[str, Any]:
        assert self.container is not None, "Container not started"

        env = dict(self.config.env)

        try:
            result = await asyncio.wait_for(
                self.container.exec(
                    command, timeout=self.config.exe_timeout, cwd=self.config.cwd,
                    login_shell=True, env=env,
                ),
                timeout=self.config.exe_timeout + 30,
            )
        except asyncio.TimeoutError:
            self.logger.error(
                "Container exec timed out (container may have been killed by the sandbox service). "
                f"command={command!r}"
            )
            return {"output": "error: container exec timed out — container may have been terminated", "returncode": 1}
        except Exception as e:
            self.logger.error(f"Container exec failed (container may have been killed): {rich_escape(str(e))}")
            return {"output": f"error: container exec failed — {e}", "returncode": 1}

        if result.error is not None:
            safe_error = rich_escape(str(result.error))
            self.logger.error(f"Error running command: {safe_error}")
            return {"output": f"error running the cmd: {result.error}", "returncode": 1}
        output = result.stdout or ""
        if result.stderr:
            output = output + result.stderr
        return {"output": output, "returncode": result.exit_code}

    async def cleanup(self):
        self.logger.info("Shutting the container down")
        sandbox_id = getattr(self.container, "sandbox_id", None) or self._sandbox_id

        deleted = False
        try:
            if self.container is not None:
                await asyncio.wait_for(self.container.delete(), timeout=self.config.cleanup_timeout)
                deleted = True
        except asyncio.TimeoutError:
            self.logger.warning("Container delete timed out (container may already be gone)")
        except Exception as e:
            self.logger.warning(f"Container delete failed (container may already be gone): {e}")

        if sandbox_id and not deleted:
            try:
                await asyncio.wait_for(self.client.delete_sandbox(sandbox_id), timeout=self.config.cleanup_timeout)
                deleted = True
            except Exception:
                pass

        if sandbox_id and not deleted:
            self.logger.warning(f"Sandbox {sandbox_id}: deletion was not confirmed")

        try:
            await asyncio.wait_for(self.client.close(cleanup=False), timeout=self.config.cleanup_timeout)
        except Exception as e:
            self.logger.warning(f"Client session close failed: {e}")

        self.container = None
        self.logger.info("Shutting the container down succeeded")

    def __del__(self):
        if self.container is None:
            return
        else:
            self.container.delete_sync()
