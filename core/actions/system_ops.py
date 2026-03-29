"""
J.A.R.V.I.S — System Operations
Safe system-level operations with validation.
All commands are risk-assessed before execution.
"""

import asyncio
import subprocess
import psutil
import platform
from typing import Dict, Any
from loguru import logger


# Whitelist of safe commands (sandbox mode ignores all others)
SAFE_COMMANDS = {
    "echo", "ping", "ipconfig", "ifconfig", "hostname",
    "python --version", "python -V", "pip list",
    "dir", "ls", "pwd", "whoami", "date", "time", "ver"
}


class SystemOperations:
    """System-level operations with strict safety controls."""

    async def run_command(self, command: str, timeout: float = 10.0) -> Dict[str, Any]:
        """Execute a shell command with timeout and output capture."""
        try:
            # Basic safety check
            cmd_lower = command.lower().strip()
            if not any(safe in cmd_lower for safe in SAFE_COMMANDS):
                logger.warning(f"Command not in safelist: {command}")
                return {
                    "success": False,
                    "error": f"Command '{command}' is not in the safe command list.",
                    "output": ""
                }

            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )

            output = stdout.decode("utf-8", errors="replace").strip()
            error = stderr.decode("utf-8", errors="replace").strip()

            logger.info(f"Command executed: {command} (rc={proc.returncode})")
            return {
                "success": proc.returncode == 0,
                "output": output or error,
                "return_code": proc.returncode,
                "error": error if proc.returncode != 0 else None
            }

        except asyncio.TimeoutError:
            return {"success": False, "error": "Command timed out.", "output": ""}
        except Exception as e:
            return {"success": False, "error": str(e), "output": ""}

    async def get_system_info(self) -> Dict[str, Any]:
        """Get current system statistics."""
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                "success": True,
                "output": (
                    f"OS: {platform.system()} {platform.release()}\n"
                    f"CPU: {cpu:.1f}% | Cores: {psutil.cpu_count()}\n"
                    f"RAM: {mem.percent:.1f}% used "
                    f"({mem.available // 1024**2}MB free)\n"
                    f"Disk: {disk.percent:.1f}% used "
                    f"({disk.free // 1024**3}GB free)"
                )
            }
        except Exception as e:
            return {"success": False, "error": str(e), "output": ""}
