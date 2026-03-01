"""
agent-safety-mcp — MCP server wrapping the AI Agent Safety Stack.

Tools exposed:
  - cost_guard_status     — Check current budget spend
  - cost_guard_check      — Pre-check if a model call is within budget
  - cost_guard_record     — Record a completed LLM call's cost
  - cost_guard_configure  — Set budget and alert threshold
  - injection_scan        — Scan text for prompt injection patterns
  - injection_check       — Scan + raise if injection detected
  - trace_start           — Start a new trace session
  - trace_step            — Log a decision step
  - trace_summary         — Get current trace summary
  - trace_save            — Save trace to disk

Install:
  pip install agent-safety-mcp

Configure in Claude Code:
  claude mcp add agent-safety -- uvx agent-safety-mcp
"""

from mcp.server.fastmcp import FastMCP

from ai_cost_guard import CostGuard, PROVIDERS
from ai_cost_guard.core.guard import BudgetExceededError
from prompt_shield import PromptScanner, InjectionRiskError
from ai_trace import Tracer

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the MCP server (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
