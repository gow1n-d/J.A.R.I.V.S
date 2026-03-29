"""
J.A.R.V.I.S — File Operations
Safe file system operations with validation and sandboxing.
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, Any
from loguru import logger


ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".json", ".yaml", ".csv", ".log",
    ".py", ".js", ".html", ".css", ".xml"
}

WORKSPACE_ROOT = Path.cwd()


class FileOperations:
    """Safe file operations within workspace boundaries."""

    def _validate_path(self, filepath: str) -> Path:
        """Ensure path is within workspace and has allowed extension."""
        path = Path(filepath)
        if not path.is_absolute():
            path = WORKSPACE_ROOT / path

        # Security: prevent directory traversal
        try:
            path.resolve().relative_to(WORKSPACE_ROOT.resolve())
        except ValueError:
            raise PermissionError(f"Path outside workspace: {filepath}")

        if path.suffix and path.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError(f"File type not allowed: {path.suffix}")

        return path

    async def create_file(self, filepath: str, content: str) -> Dict[str, Any]:
        """Create a file with given content."""
        try:
            path = self._validate_path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            logger.info(f"Created file: {path}")
            return {
                "success": True,
                "output": f"File created: {path.name} ({path.stat().st_size} bytes)",
                "path": str(path)
            }
        except Exception as e:
            logger.error(f"File create failed: {e}")
            return {"success": False, "error": str(e), "output": ""}

    async def read_file(self, filepath: str) -> Dict[str, Any]:
        """Read a file's contents."""
        try:
            path = self._validate_path(filepath)
            if not path.exists():
                return {"success": False, "error": f"File not found: {filepath}", "output": ""}
            content = path.read_text(encoding="utf-8", errors="replace")
            logger.info(f"Read file: {path}")
            return {
                "success": True,
                "output": content[:2000],  # Limit output size
                "path": str(path),
                "size": len(content)
            }
        except Exception as e:
            return {"success": False, "error": str(e), "output": ""}

    async def append_file(self, filepath: str, content: str) -> Dict[str, Any]:
        """Append content to existing file."""
        try:
            path = self._validate_path(filepath)
            with open(path, "a", encoding="utf-8") as f:
                f.write(content)
            return {"success": True, "output": f"Appended to {path.name}", "path": str(path)}
        except Exception as e:
            return {"success": False, "error": str(e), "output": ""}

    async def list_directory(self, dirpath: str = ".") -> Dict[str, Any]:
        """List contents of a directory."""
        try:
            path = WORKSPACE_ROOT / dirpath
            items = []
            for item in sorted(path.iterdir()):
                items.append({
                    "name": item.name,
                    "type": "dir" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else 0
                })
            output = "\n".join(
                f"{'📁' if i['type']=='dir' else '📄'} {i['name']} "
                f"({'dir' if i['type']=='dir' else str(i['size'])+' bytes'})"
                for i in items
            )
            return {"success": True, "output": output, "items": items}
        except Exception as e:
            return {"success": False, "error": str(e), "output": ""}
