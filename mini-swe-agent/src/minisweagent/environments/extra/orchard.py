import logging
import os
from typing import Any

from pydantic import BaseModel

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge


class OrchardEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = {}
    """Environment variables to set in the sandbox."""
    forward_env: list[str] = []
    """Environment variables to forward from the host to the sandbox."""
    timeout: int = 30
    """Timeout for executing commands in the sandbox."""
    base_url: str = os.getenv("SANDBOX_BASE_URL", "http://localhost:8000")
    """Base URL of the Orchard sandbox service."""
    api_key: str = os.getenv("SANDBOX_API_KEY", "")
    """API key for authenticating with the Orchard sandbox service."""
    block_network: bool = True
    """Whether to block network egress in the sandbox."""
    cpu: str | None = None
    """CPU request/limit (e.g., '4', '2000m')."""
    memory: str | None = None
    """Memory request/limit (e.g., '16Gi')."""
    sandbox_timeout: int = 3600
    """Timeout in seconds for sandbox to become ready."""
    login_shell: bool = True
    """Whether to use login shell (bash -lc) for command execution."""
    heartbeat_interval: int = 60
    """Interval in seconds between heartbeats to keep sandbox alive. 0 to disable."""


class OrchardEnvironment:
    def __init__(
        self,
        *,
        config_class: type = OrchardEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        """Execute commands in a remote Orchard sandbox via the `orchard_env` client.
        See `OrchardEnvironmentConfig` for keyword arguments.
        """
        from orchard_env import SandboxClient

        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)
        self._client = SandboxClient(
            base_url=self.config.base_url,
            api_key=self.config.api_key or None,
        )
        self._sandbox = self._client.create_sandbox(
            image=self.config.image,
            block_network=self.config.block_network,
            cpu=self.config.cpu,
            memory=self.config.memory,
            timeout=self.config.sandbox_timeout,
        )
        self.logger.info(f"Created sandbox {self._sandbox.sandbox_id}")
        if self.config.heartbeat_interval > 0:
            self._sandbox.start_heartbeat(interval=self.config.heartbeat_interval)

        bashrc_command = f"echo 'cd {self.config.cwd}' >> /root/.bashrc"
        self._sandbox.exec(
            command=bashrc_command,
            timeout=self.config.timeout,
            cwd=self.config.cwd,
            login_shell=True,
            env=None,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def execute(self, action: dict, cwd: str = "/", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the sandbox and return the result as a dict."""
        command = action.get("command", "")
        cwd = self.config.cwd if cwd == '/' else cwd
        exec_timeout = timeout or self.config.timeout

        env: dict[str, str] = {}
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                env[key] = value
        env.update(self.config.env)

        try:
            result = self._sandbox.exec(
                command=command,
                timeout=exec_timeout,
                cwd=cwd,
                env=env or None,
                login_shell=self.config.login_shell,
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            combined = stdout + stderr if stderr else stdout
            output = {
                "output": combined,
                "returncode": result.exit_code if result.exit_code is not None else (-1 if result.failed else 0),
                "exception_info": result.error or "",
            }
        except Exception as e:
            output = {
                "output": "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }

        self._check_finished(output)
        return output

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def cleanup(self):
        """Delete the sandbox and close the client."""
        if getattr(self, "_cleaned_up", False):
            return
        self._cleaned_up = True
        if hasattr(self, "_sandbox"):
            try:
                self._sandbox.delete()
                self.logger.info(f"Deleted sandbox {self._sandbox.sandbox_id}")
            except Exception:
                pass
        if hasattr(self, "_client"):
            self._client.close()

    def __del__(self):
        self.cleanup()
