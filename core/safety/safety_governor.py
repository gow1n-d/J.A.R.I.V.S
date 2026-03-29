"""
J.A.R.V.I.S — Safety Governor (IMMUTABLE CORE)
The absolute safety layer. Cannot be modified or bypassed.
Every action passes through here before execution.

IMMUTABLE CONTRACT:
- User always has full control
- No critical actions without confirmation
- Complete transparency in all operations
- Sandbox mode enforced by default
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from loguru import logger


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyResult:
    """Result of safety evaluation."""
    approved: bool
    risk_level: str
    requires_confirmation: bool
    blocked_reason: Optional[str]
    warnings: List[str]
    safe_alternative: Optional[str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class SafetyLog:
    """Immutable safety audit log entry."""
    action_type: str
    risk_level: str
    approved: bool
    reason: str
    timestamp: float = field(default_factory=time.time)


class RiskClassifier:
    """Classifies action risk levels."""

    # Absolute blocks — must be explicit destructive shell/SQL commands
    # Keep these specific to avoid false positives on normal conversation
    BLOCKED_PATTERNS = [
        "rm -rf /",
        "del /s /q c:\\",
        "format c:",
        "DROP TABLE",
        "DELETE FROM",
        "__import__('os').system",
        "subprocess.call(['rm",
        "os.system('rm",
        "shutil.rmtree('/')",
    ]

    # High risk — always require confirmation
    HIGH_RISK_TYPES = {
        "delete_action", "execute_destructive",
        "system_format", "registry_edit"
    }

    # Medium risk — require confirmation unless sandbox
    MEDIUM_RISK_TYPES = {
        "execute_action", "system_op", "file_delete",
        "create_action", "plan_action"
    }

    # Low risk — auto-approve
    LOW_RISK_TYPES = {
        "respond", "analyze", "generate_help",
        "chat_response", "read", "query", "status"
    }

    def classify(self, action_choice) -> str:
        action_type = action_choice.action_type

        # Check action-level risk override
        if hasattr(action_choice, 'risk_level'):
            return action_choice.risk_level

        if action_type in self.HIGH_RISK_TYPES:
            return RiskLevel.HIGH.value
        elif action_type in self.MEDIUM_RISK_TYPES:
            return RiskLevel.MEDIUM.value
        elif action_type in self.LOW_RISK_TYPES:
            return RiskLevel.LOW.value

        return RiskLevel.MEDIUM.value

    def has_blocked_pattern(self, text: str) -> bool:
        """Only check explicit user-supplied text, not Python object representations."""
        # Limit check to first 500 chars of cleaned text to avoid matching Python internals
        text_lower = str(text)[:500].lower()
        return any(
            pattern.lower() in text_lower
            for pattern in self.BLOCKED_PATTERNS
        )


class PermissionSystem:
    """
    Manages action permissions and user consent tracking.
    Remembers granted and denied permissions across sessions.
    """

    def __init__(self):
        self._granted: Dict[str, float] = {}     # action_type → expiry timestamp
        self._denied: Dict[str, float] = {}       # action_type → denial timestamp
        self._session_grants: List[str] = []

    def is_granted(self, action_type: str) -> bool:
        if action_type in self._granted:
            if self._granted[action_type] > time.time():
                return True
            else:
                del self._granted[action_type]
        return False

    def grant(self, action_type: str, duration_seconds: float = 300):
        """Grant permission for an action type for duration."""
        self._granted[action_type] = time.time() + duration_seconds
        self._session_grants.append(action_type)

    def deny(self, action_type: str):
        self._denied[action_type] = time.time()

    def is_denied(self, action_type: str) -> bool:
        # Denials expire after 60 seconds (user can retry)
        if action_type in self._denied:
            if time.time() - self._denied[action_type] < 60:
                return True
        return False


class SafetyGovernor:
    """
    IMMUTABLE SAFETY LAYER.
    This class cannot be modified at runtime.
    All actions MUST pass through evaluate() before execution.
    """

    def __init__(self, config: dict):
        self.config = config
        self.sandbox_mode = config.get("sandbox_mode", True)
        self.max_risk_level = config.get("max_risk_level", "medium")
        self.require_confirmation_above = config.get(
            "require_confirmation_above", "low"
        )
        self.log_all_actions = config.get("log_all_actions", True)

        self.risk_classifier = RiskClassifier()
        self.permissions = PermissionSystem()

        self._audit_log: List[SafetyLog] = []
        self._blocked_count = 0
        self._approved_count = 0

        self._active = True  # Cannot be set to False externally
        logger.info("Safety Governor armed. IMMUTABLE MODE ACTIVE.")

    async def evaluate(self, action_choice) -> SafetyResult:
        """
        Evaluate an action through ALL safety checks.
        Returns SafetyResult with approval status.
        """
        risk_level = self.risk_classifier.classify(action_choice)
        warnings = []

        # 1. ABSOLUTE BLOCK: Check raw user input only — NOT the entire serialized task graph
        # Scanning str(parameters) causes false positives because Python object
        # repr() contains keywords like 'format', 'del', etc. from internal code.
        raw_input = ""
        params = getattr(action_choice, 'parameters', {})
        if isinstance(params, dict):
            raw_input = str(params.get('raw_input', '') or params.get('goal_description', ''))
        if self.risk_classifier.has_blocked_pattern(raw_input):
            result = SafetyResult(
                approved=False,
                risk_level=RiskLevel.CRITICAL.value,
                requires_confirmation=False,
                blocked_reason="⛔ BLOCKED: Dangerous pattern detected in action parameters.",
                warnings=[],
                safe_alternative="Please rephrase the request without destructive operations."
            )
            self._log(action_choice.action_type, RiskLevel.CRITICAL.value, False,
                      "Dangerous pattern detected")
            self._blocked_count += 1
            return result

        # 2. HALT action always approved (user stop)
        if action_choice.action_type == "halt":
            return SafetyResult(
                approved=True,
                risk_level=RiskLevel.LOW.value,
                requires_confirmation=False,
                blocked_reason=None,
                warnings=[],
                safe_alternative=None
            )

        # 3. Risk level gate
        risk_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        max_allowed = risk_order.get(self.max_risk_level, 1)
        action_risk = risk_order.get(risk_level, 1)

        if action_risk > max_allowed:
            result = SafetyResult(
                approved=False,
                risk_level=risk_level,
                requires_confirmation=False,
                blocked_reason=(
                    f"⛔ Action risk ({risk_level}) exceeds configured maximum ({self.max_risk_level}). "
                    f"Modify config.yaml to allow higher risk actions."
                ),
                warnings=[f"Risk level {risk_level} blocked by safety governor."],
                safe_alternative="Consider a lower-risk alternative approach."
            )
            self._blocked_count += 1
            self._log(action_choice.action_type, risk_level, False, "Risk too high")
            return result

        # 4. Confirmation requirements
        confirm_threshold = risk_order.get(self.require_confirmation_above, 0)
        requires_confirmation = action_risk > confirm_threshold

        if requires_confirmation and not self.permissions.is_granted(action_choice.action_type):
            # In CLI mode, request confirmation
            confirmed = await self._request_confirmation(action_choice, risk_level)
            if not confirmed:
                result = SafetyResult(
                    approved=False,
                    risk_level=risk_level,
                    requires_confirmation=True,
                    blocked_reason="Action denied by user.",
                    warnings=["User declined to authorize action."],
                    safe_alternative=None
                )
                self.permissions.deny(action_choice.action_type)
                self._blocked_count += 1
                self._log(action_choice.action_type, risk_level, False, "User denied")
                return result

        # 5. Sandbox enforcement
        if self.sandbox_mode and risk_level in {"high", "critical"}:
            warnings.append("⚠ Sandbox mode: High-risk action simulated, not executed.")

        # APPROVED
        result = SafetyResult(
            approved=True,
            risk_level=risk_level,
            requires_confirmation=requires_confirmation,
            blocked_reason=None,
            warnings=warnings,
            safe_alternative=None
        )

        self._approved_count += 1
        self._log(action_choice.action_type, risk_level, True, "Approved")
        return result

    async def _request_confirmation(self, action_choice, risk_level: str) -> bool:
        """Interactive confirmation for medium/high risk actions."""
        from rich.console import Console
        console = Console()
        try:
            console.print(
                f"\n[bold yellow]⚠ CONFIRMATION REQUIRED[/bold yellow]\n"
                f"Action: [cyan]{action_choice.action_type}[/cyan]\n"
                f"Risk Level: [red]{risk_level.upper()}[/red]\n"
                f"Rationale: {getattr(action_choice, 'rationale', 'N/A')}"
            )
            response = console.input("[bold]Authorize? (yes/no): [/bold]").strip().lower()
            confirmed = response in {"yes", "y", "ok", "sure", "proceed"}
            if confirmed:
                # Grant for 5 minutes
                self.permissions.grant(action_choice.action_type, 300)
            return confirmed
        except Exception:
            return False  # Fail safe: deny on error

    def _log(self, action_type: str, risk: str, approved: bool, reason: str):
        """Immutable audit log entry."""
        entry = SafetyLog(
            action_type=action_type,
            risk_level=risk,
            approved=approved,
            reason=reason
        )
        self._audit_log.append(entry)
        # Keep last 10000 entries
        if len(self._audit_log) > 10000:
            self._audit_log.pop(0)
        logger.debug(f"Safety: {action_type} | {risk} | {'✓' if approved else '✗'} | {reason}")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "approved": self._approved_count,
            "blocked": self._blocked_count,
            "sandbox_mode": self.sandbox_mode,
            "audit_entries": len(self._audit_log),
            "max_risk": self.max_risk_level
        }

    def get_audit_log(self, n: int = 20) -> List[Dict]:
        return [
            {"action": e.action_type, "risk": e.risk_level,
             "approved": e.approved, "reason": e.reason, "time": e.timestamp}
            for e in self._audit_log[-n:]
        ]
