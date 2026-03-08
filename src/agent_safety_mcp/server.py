"""
agent-safety-mcp v0.2.0 — MCP server wrapping the AI Agent Safety Stack.

Tools exposed:
  Cost Guard:
  - cost_guard_configure  — Set budget and alert threshold
  - cost_guard_status     — Check current budget spend
  - cost_guard_check      — Pre-check if a model call is within budget
  - cost_guard_record     — Record a completed LLM call's cost
  - cost_guard_models     — List supported models with pricing

  Injection Guard:
  - injection_scan        — Scan text for prompt injection patterns
  - injection_check       — Scan + raise if injection detected
  - injection_patterns    — List all detection patterns

  Decision Tracer:
  - trace_start           — Start a new trace session
  - trace_step            — Log a decision step
  - trace_summary         — Get current trace summary
  - trace_save            — Save trace to disk

  KYA Identity (v0.2.0):
  - kya_create_card       — Create an agent identity card
  - kya_sign_card         — Sign a card with Ed25519
  - kya_verify_card       — Verify card structure and signature
  - kya_generate_keypair  — Generate Ed25519 signing keypair

  Unified (v0.2.0):
  - safety_check          — Run injection scan + cost check + trace in one call

Install:
  pip install agent-safety-mcp

Configure in Claude Code:
  claude mcp add agent-safety -- uvx agent-safety-mcp
"""

import json
from mcp.server.fastmcp import FastMCP

from ai_cost_guard import CostGuard, PROVIDERS
from ai_cost_guard.core.guard import BudgetExceededError
from prompt_shield import PromptScanner, InjectionRiskError
from ai_trace import Tracer
import tempfile
import os
from kya.signer import generate_keypair, sign_card, verify_card
from kya.validator import validate, compute_completeness_score

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "Agent Safety",
    instructions=(
        "AI agent safety tools: cost budget enforcement, prompt injection "
        "scanning, and decision tracing. Use these tools to protect LLM "
        "applications from cost blowouts, adversarial inputs, and "
        "untraceable failures."
    ),
)

# ---------------------------------------------------------------------------
# Shared state (persists across tool calls within a session)
# ---------------------------------------------------------------------------
_guard: CostGuard | None = None
_scanner: PromptScanner | None = None
_tracer: Tracer | None = None


def _get_guard() -> CostGuard:
    global _guard
    if _guard is None:
        _guard = CostGuard(weekly_budget_usd=10.00)
    return _guard


def _get_scanner() -> PromptScanner:
    global _scanner
    if _scanner is None:
        _scanner = PromptScanner(threshold="MEDIUM")
    return _scanner


def _get_tracer(agent: str = "default") -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = Tracer(agent=agent, trace_dir="./traces")
    return _tracer


# =========================================================================
#  COST GUARD TOOLS
# =========================================================================

@mcp.tool()
def cost_guard_configure(
    weekly_budget_usd: float = 10.0,
    alert_at_pct: float = 0.80,
    dry_run: bool = False,
) -> dict:
    """Configure the cost guard budget.

    Args:
        weekly_budget_usd: Maximum spend per week in USD.
        alert_at_pct: Warn when spend reaches this percentage (0.0-1.0).
        dry_run: If true, all calls raise BudgetExceededError (safe for testing).
    """
    global _guard
    _guard = CostGuard(
        weekly_budget_usd=weekly_budget_usd,
        alert_at_pct=alert_at_pct,
        dry_run=dry_run,
    )
    return {
        "status": "configured",
        "weekly_budget_usd": weekly_budget_usd,
        "alert_at_pct": alert_at_pct,
        "dry_run": dry_run,
    }


@mcp.tool()
def cost_guard_status() -> dict:
    """Check current budget spend — how much is left, percentage used, call details."""
    return _get_guard().status()


@mcp.tool()
def cost_guard_check(
    model: str,
    estimated_input_tokens: int = 1000,
    estimated_output_tokens: int = 500,
) -> dict:
    """Pre-check if a model call is within budget. Returns safe/blocked status.

    Args:
        model: Model identifier (e.g. "anthropic/claude-haiku-4-5-20251001", "openai/gpt-4o").
        estimated_input_tokens: Expected input token count.
        estimated_output_tokens: Expected output token count.
    """
    guard = _get_guard()
    try:
        guard.check_budget(model, estimated_input_tokens, estimated_output_tokens)
        status = guard.status()
        return {
            "allowed": True,
            "model": model,
            "estimated_cost_usd": round(
                estimated_input_tokens * PROVIDERS.get(model, {}).get("input", 0)
                + estimated_output_tokens * PROVIDERS.get(model, {}).get("output", 0),
                6,
            ),
            "remaining_usd": status.get("remaining_usd"),
        }
    except BudgetExceededError as e:
        return {
            "allowed": False,
            "model": model,
            "reason": str(e),
        }


@mcp.tool()
def cost_guard_record(
    model: str,
    input_tokens: int,
    output_tokens: int,
    purpose: str = "",
) -> dict:
    """Record a completed LLM call's token usage and cost.

    Args:
        model: Model identifier used for the call.
        input_tokens: Actual input tokens consumed.
        output_tokens: Actual output tokens consumed.
        purpose: Optional label for this call (e.g. "summarizer", "classifier").
    """
    guard = _get_guard()
    cost = guard.record(model, input_tokens, output_tokens, purpose=purpose)
    return {
        "recorded": True,
        "model": model,
        "cost_usd": round(cost, 6),
        "total_spent_usd": round(guard.status().get("spent_usd", 0), 6),
    }


@mcp.tool()
def cost_guard_models() -> dict:
    """List all supported models with their per-token pricing."""
    models = {}
    for name, pricing in PROVIDERS.items():
        models[name] = {
            "input_per_1M": round(pricing["input"] * 1_000_000, 2),
            "output_per_1M": round(pricing["output"] * 1_000_000, 2),
        }
    return models


# =========================================================================
#  INJECTION GUARD TOOLS
# =========================================================================

@mcp.tool()
def injection_scan(text: str, threshold: str = "MEDIUM") -> dict:
    """Scan text for prompt injection patterns. Returns risk assessment without blocking.

    Args:
        text: The text to scan for injection attempts.
        threshold: Sensitivity level — "LOW", "MEDIUM", "HIGH", or "CRITICAL".
    """
    scanner = PromptScanner(threshold=threshold)
    result = scanner.scan(text)
    return {
        "severity": result.severity,
        "risk_score": result.risk_score,
        "is_safe": result.is_safe,
        "matches": result.matches,
        "text_preview": result.text[:200],
    }


@mcp.tool()
def injection_check(text: str, threshold: str = "MEDIUM") -> dict:
    """Scan text and block if injection is detected above threshold.

    Args:
        text: The text to check for injection attempts.
        threshold: Block at this severity or above — "LOW", "MEDIUM", "HIGH", "CRITICAL".
    """
    scanner = PromptScanner(threshold=threshold)
    try:
        result = scanner.check(text)
        return {
            "allowed": True,
            "severity": result.severity,
            "risk_score": result.risk_score,
        }
    except InjectionRiskError as e:
        return {
            "allowed": False,
            "severity": e.severity,
            "risk_score": e.risk_score,
            "matches": e.matches,
            "text_preview": e.text[:200],
        }


@mcp.tool()
def injection_patterns() -> list[dict]:
    """List all built-in injection detection patterns with categories and weights."""
    from prompt_shield import PATTERNS
    return [
        {"name": p["name"], "category": p["category"], "weight": p["weight"]}
        for p in PATTERNS
    ]


# =========================================================================
#  TRACE TOOLS
# =========================================================================

@mcp.tool()
def trace_start(
    agent: str = "default",
    trace_dir: str = "./traces",
    model: str = "",
) -> dict:
    """Start a new trace session for an AI agent.

    Args:
        agent: Agent name (used in filenames).
        trace_dir: Directory to save trace files.
        model: Optional model name to attach as metadata.
    """
    global _tracer
    meta = {}
    if model:
        meta["model"] = model
    _tracer = Tracer(agent=agent, trace_dir=trace_dir, meta=meta if meta else None)
    return {
        "status": "started",
        "agent": agent,
        "trace_dir": trace_dir,
    }


@mcp.tool()
def trace_step(
    name: str,
    decision: str = "",
    confidence: float = 0.0,
    input_data: str = "",
    reason: str = "",
    outcome: str = "ok",
) -> dict:
    """Log a decision step in the current trace session.

    Args:
        name: Step name (e.g. "analyze_signal", "classify_intent").
        decision: What the agent decided.
        confidence: Confidence score (0.0-1.0).
        input_data: What the agent saw (brief description).
        reason: Why this decision was made.
        outcome: "ok" or "error".
    """
    tracer = _get_tracer()
    with tracer.step(name, input=input_data) as step:
        logs = {}
        if decision:
            logs["decision"] = decision
        if confidence:
            logs["confidence"] = confidence
        if reason:
            logs["reason"] = reason
        if logs:
            step.log(**logs)
        if outcome == "error":
            step.fail(reason=reason or "unspecified error")

    return {
        "recorded": True,
        "step": name,
        "outcome": outcome,
        "duration_ms": step.duration_ms,
    }


@mcp.tool()
def trace_summary() -> dict:
    """Get a summary of the current trace session — step count, errors, timing."""
    tracer = _get_tracer()
    return tracer.summary()


@mcp.tool()
def trace_save() -> dict:
    """Save the current trace to disk as JSON and Markdown files."""
    tracer = _get_tracer()
    json_path = tracer.save()
    md_path = tracer.save_markdown()
    return {
        "saved": True,
        "json": str(json_path),
        "markdown": str(md_path),
        "summary": tracer.summary(),
    }


# =========================================================================
#  KYA IDENTITY TOOLS (v0.2.0)
# =========================================================================

_kya_cards: dict[str, dict] = {}
_kya_key_name: str | None = None


def _card_to_file(card: dict) -> str:
    """Write a card dict to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".json", prefix="kya_card_")
    with os.fdopen(fd, "w") as f:
        json.dump(card, f)
    return path


@mcp.tool()
def kya_generate_keypair(name: str = "mcp-session") -> dict:
    """Generate an Ed25519 keypair for signing agent identity cards.

    Args:
        name: Key name (default "mcp-session"). Keys stored at ~/.kya/keys/
    """
    global _kya_key_name
    try:
        priv_path, pub_path = generate_keypair(name)
        _kya_key_name = name
        return {
            "status": "generated",
            "private_key": str(priv_path),
            "public_key": str(pub_path),
            "note": "Use kya_sign_card to sign cards with this key.",
        }
    except Exception as e:
        return {"error": f"Failed to generate keypair: {e}. Install cryptography: pip install cryptography"}


@mcp.tool()
def kya_create_card(
    agent_id: str,
    name: str = "",
    purpose: str = "",
    capabilities: str = "",
    owner_name: str = "",
    version: str = "0.1.0",
) -> dict:
    """Create a KYA (Know Your Agent) identity card for an agent.

    Args:
        agent_id: Unique ID in format "org/agent-name" (e.g. "luciferforge/research-bot").
        name: Human-readable agent name.
        purpose: What this agent does (min 10 chars for validity).
        capabilities: Comma-separated list of capabilities (e.g. "text_generation,web_search").
        owner_name: Owner/organization name.
        version: Agent version string.
    """
    caps_list = [c.strip() for c in capabilities.split(",") if c.strip()] if capabilities else []
    caps_obj = {c: {"description": c, "risk_level": "low"} for c in caps_list}

    card = {
        "kya_version": "0.2",
        "agent_id": agent_id,
        "name": name or agent_id.split("/")[-1],
        "version": version,
        "purpose": purpose,
        "capabilities": caps_obj,
        "owner": {"name": owner_name or "unknown"},
    }

    _kya_cards[agent_id] = card
    score = compute_completeness_score(card)

    return {
        "status": "created",
        "agent_id": agent_id,
        "completeness_score": score,
        "card": card,
    }


@mcp.tool()
def kya_sign_card(agent_id: str) -> dict:
    """Sign an existing KYA card with the session's Ed25519 private key.

    Args:
        agent_id: The agent_id of the card to sign. Must call kya_create_card first.
    """
    if agent_id not in _kya_cards:
        return {"error": f"No card found for agent_id '{agent_id}'. Create one first."}

    if _kya_key_name is None:
        return {"error": "No keypair generated. Call kya_generate_keypair first."}

    from kya.signer import KEY_DIR
    priv_path = str(KEY_DIR / f"{_kya_key_name}.key")
    if not os.path.exists(priv_path):
        return {"error": f"Private key not found at {priv_path}"}

    try:
        signed = sign_card(_kya_cards[agent_id], priv_path)
        _kya_cards[agent_id] = signed
        return {
            "status": "signed",
            "agent_id": agent_id,
            "has_signature": "_signature" in signed or "signature" in signed,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def kya_verify_card(agent_id: str = "", card_json: str = "") -> dict:
    """Verify a KYA identity card — check structure, completeness, and signature.

    Args:
        agent_id: Look up a card created in this session by agent_id.
        card_json: Or pass raw card JSON to verify an external card.
    """
    if card_json:
        try:
            card = json.loads(card_json)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON"}
    elif agent_id and agent_id in _kya_cards:
        card = _kya_cards[agent_id]
    else:
        return {"error": "Provide agent_id (from this session) or card_json"}

    # Write card to temp file for validate()
    card_path = _card_to_file(card)
    try:
        validation = validate(card_path)
    finally:
        os.unlink(card_path)

    score = compute_completeness_score(card)

    result = {
        "valid": validation.get("valid", False),
        "errors": validation.get("errors", []),
        "completeness_score": score,
        "agent_id": card.get("agent_id", "unknown"),
        "has_signature": "_signature" in card or "signature" in card,
    }

    if ("_signature" in card or "signature" in card) and _kya_key_name:
        from kya.signer import KEY_DIR
        pub_path = str(KEY_DIR / f"{_kya_key_name}.pub")
        if os.path.exists(pub_path):
            try:
                verified = verify_card(card, pub_path)
                result["signature_verified"] = verified.get("valid", False)
            except Exception:
                result["signature_verified"] = False

    return result


# =========================================================================
#  UNIFIED SAFETY CHECK (v0.2.0)
# =========================================================================

@mcp.tool()
def safety_check(
    text: str,
    model: str = "",
    estimated_input_tokens: int = 0,
    estimated_output_tokens: int = 0,
    step_name: str = "safety_check",
) -> dict:
    """Run a unified safety check: injection scan + cost check + trace step.

    This is the recommended single tool for pre-flight safety. It runs
    injection scanning, checks the cost budget, and logs the decision.

    Args:
        text: The input text to scan for injections.
        model: Model identifier for cost checking (optional).
        estimated_input_tokens: Expected input tokens for cost check.
        estimated_output_tokens: Expected output tokens for cost check.
        step_name: Name for the trace step.
    """
    results: dict = {"safe": True, "checks": {}}

    # 1. Injection scan
    scanner = _get_scanner()
    scan = scanner.scan(text)
    results["checks"]["injection"] = {
        "is_safe": scan.is_safe,
        "risk_score": scan.risk_score,
        "severity": scan.severity,
        "matches": [m.get("name", "") for m in scan.matches],
    }
    if not scan.is_safe:
        results["safe"] = False
        results["blocked_by"] = "injection"

    # 2. Cost check (if model provided)
    if model and (estimated_input_tokens or estimated_output_tokens):
        guard = _get_guard()
        try:
            guard.check_budget(model, estimated_input_tokens, estimated_output_tokens)
            results["checks"]["cost"] = {"allowed": True}
        except BudgetExceededError as e:
            results["safe"] = False
            results["blocked_by"] = results.get("blocked_by", "cost")
            results["checks"]["cost"] = {"allowed": False, "reason": str(e)}

    # 3. Trace
    tracer = _get_tracer()
    action = "allowed" if results["safe"] else "blocked"
    with tracer.step(step_name, action=action, **results["checks"]):
        pass

    results["checks"]["trace"] = {"recorded": True}

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the MCP server (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
