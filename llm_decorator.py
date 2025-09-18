# oitrace/llm_decorator.py
from __future__ import annotations
import asyncio, json, math, time
from typing import Any, Callable, Dict, Optional, Tuple
from functools import wraps
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

# OpenInference semantic conventions
from openinference.semconv.trace import (
    SpanAttributes,
    OpenInferenceSpanKindValues,
)

Json = Any

# ---------------------------
# Helpers
# ---------------------------

def _shorten(s: str, limit: int = 4000) -> str:
    if not isinstance(s, str):
        s = json.dumps(s, ensure_ascii=False, default=str)
    if len(s) <= limit:
        return s
    head = math.floor(limit * 0.7)
    tail = limit - head - 3
    return f"{s[:head]}...{s[-tail:]}"


def _json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return _shorten(repr(obj))


# ---------------------------
# Default extractors for OpenAI-style requests/responses
# ---------------------------

def default_request_extractor(fn, args, kwargs) -> Dict[str, Any]:
    """
    Pulls common Chat Completions params from typical SDK signatures.
    Looks for: messages, model, and common invocation params.
    """
    # Guess common param names
    messages = kwargs.get("messages") or kwargs.get("input") or kwargs.get("inputs")
    model = kwargs.get("model")
    params = {
        k: v
        for k, v in kwargs.items()
        if k in {
            "temperature", "top_p", "max_tokens", "stop", "frequency_penalty",
            "presence_penalty", "seed", "response_format", "tools", "tool_choice",
        }
    }
    return {"messages": messages, "model": model, "params": params}


def default_response_extractor(resp: Any) -> Dict[str, Any]:
    """
    Extracts model, messages, token usage from an OpenAI-compatible response.
    Supports dicts or simple SDK objects with dotted attributes.
    """
    # Support dict or dot-attrs
    asdict = resp if isinstance(resp, dict) else getattr(resp, "model_dump", lambda: None)() or {}
    if not asdict:
        # fall back to attribute probing
        asdict = {
            "model": getattr(resp, "model", None),
            "choices": getattr(resp, "choices", None),
            "usage": getattr(resp, "usage", None),
        }

    model = asdict.get("model")
    usage = asdict.get("usage") or {}
    # Standard OpenAI usage keys
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    # OpenAI “prompt_tokens_details.cached_tokens” → llm.token_count.prompt_details.cache_read
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached = prompt_details.get("cached_tokens")

    # Collect assistant messages (first choice is fine; extend if needed)
    choices = asdict.get("choices") or []
    out_msgs = []
    if choices:
        ch0 = choices[0] or {}
        # OpenAI: choices[0].message = {role, content, tool_calls?}
        msg = ch0.get("message") or {}
        out_msgs = [msg] if msg else []

    return {
        "model": model,
        "output_messages": out_msgs,
        "usage": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
            "prompt_cache_read": cached,
        },
    }


# ---------------------------
# The decorator
# ---------------------------

def trace_llm(
    *,
    tracer: Optional[trace.Tracer] = None,
    span_name: Optional[str] = None,
    system: Optional[str] = "openai",      # e.g., "openai", "anthropic", "mistralai"
    provider: Optional[str] = None,        # e.g., "azure", "aws", "google"
    # Input capture
    request_extractor: Callable[[Callable, Tuple[Any, ...], Dict[str, Any]], Dict[str, Any]] = default_request_extractor,
    # Output capture
    response_extractor: Callable[[Any], Dict[str, Any]] = default_response_extractor,
    # Optional: cost calculator if you want to emit llm.cost.*
    cost_calculator: Optional[Callable[[Dict[str, Any]], Dict[str, float]]] = None,
    input_mime: str = "application/json",
    output_mime: str = "application/json",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorate a function that performs an LLM (chat completions) call.

    Sets OpenInference LLM semantic-convention attributes:
      - openinference.span.kind = LLM
      - llm.system / llm.provider / llm.model_name
      - llm.invocation_parameters (JSON)
      - llm.input_messages / llm.output_messages (JSON)
      - llm.token_count.prompt / .completion / .total (+ cache_read when available)
      - llm.cost.* (if cost_calculator provided)
      - Also sets input/output mime/value for quick inspection
    """
    tracer = tracer or trace.get_tracer(__name__)

    def _apply_llm_attrs(span, *, req: Dict[str, Any], res: Optional[Dict[str, Any]] = None):
        # Mark as LLM span
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)

        # System/provider/model
        if system:
            span.set_attribute(SpanAttributes.LLM_SYSTEM, system)      # e.g., "openai"
        if provider:
            span.set_attribute(SpanAttributes.LLM_PROVIDER, provider)  # e.g., "azure"

        # Request: messages + invocation params
        messages = req.get("messages")
        model = req.get("model")
        params = req.get("params") or {}

        if model:
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model)
        if messages is not None:
            span.set_attribute(SpanAttributes.LLM_INPUT_MESSAGES, _json(messages))
        if params:
            span.set_attribute(SpanAttributes.LLM_INVOCATION_PARAMETERS, _json(params))

        # Also keep generic input.* for convenience
        if messages is not None:
            span.set_attribute(SpanAttributes.INPUT_VALUE, _json(messages))
            span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, input_mime)

        # Response fields (if we have them)
        if res:
            out_msgs = res.get("output_messages")
            if out_msgs is not None:
                span.set_attribute(SpanAttributes.LLM_OUTPUT_MESSAGES, _json(out_msgs))
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, _json(out_msgs))
                span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, output_mime)

            # Prefer server-returned model name if present
            resp_model = res.get("model")
            if resp_model:
                span.set_attribute(SpanAttributes.LLM_MODEL_NAME, resp_model)

            usage = res.get("usage") or {}
            if usage.get("prompt") is not None:
                span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, int(usage["prompt"]))
            if usage.get("completion") is not None:
                span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, int(usage["completion"]))
            if usage.get("total") is not None:
                span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, int(usage["total"]))
            if usage.get("prompt_cache_read") is not None:
                # maps OpenAI cached_tokens → prompt_details.cache_read per spec
                span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ, int(usage["prompt_cache_read"]))

            # Optional cost emission
            if cost_calculator:
                try:
                    costs = cost_calculator({"req": req, "res": res, "usage": usage}) or {}
                    for k in ("prompt", "completion", "total"):
                        v = costs.get(k)
                        if v is not None:
                            if k == "prompt":
                                span.set_attribute(SpanAttributes.LLM_COST_PROMPT, float(v))
                            elif k == "completion":
                                span.set_attribute(SpanAttributes.LLM_COST_COMPLETION, float(v))
                            elif k == "total":
                                span.set_attribute(SpanAttributes.LLM_COST_TOTAL, float(v))
                    # details (optional)
                    cd = costs.get("completion_details") or {}
                    if "output" in cd:
                        span.set_attribute(SpanAttributes.LLM_COST_COMPLETION_DETAILS_OUTPUT, float(cd["output"]))
                    if "reasoning" in cd:
                        span.set_attribute(SpanAttributes.LLM_COST_COMPLETION_DETAILS_REASONING, float(cd["reasoning"]))
                    if "audio" in cd:
                        span.set_attribute(SpanAttributes.LLM_COST_COMPLETION_DETAILS_AUDIO, float(cd["audio"]))
                except Exception:
                    pass

    def _decorate_sync(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            req = request_extractor(fn, args, kwargs) or {}
            _span_name = span_name or fn.__qualname__
            with tracer.start_as_current_span(_span_name, kind=SpanKind.CLIENT) as span:
                try:
                    res = fn(*args, **kwargs)
                    res_info = response_extractor(res) if res is not None else {}
                    _apply_llm_attrs(span, req=req, res=res_info)
                    span.set_status(Status(StatusCode.OK))
                    return res
                except BaseException as e:
                    # Attach request context even on failure
                    _apply_llm_attrs(span, req=req, res=None)
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        return wrapper

    def _decorate_async(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            req = request_extractor(fn, args, kwargs) or {}
            _span_name = span_name or fn.__qualname__
            with tracer.start_as_current_span(_span_name, kind=SpanKind.CLIENT) as span:
                try:
                    res = await fn(*args, **kwargs)
                    res_info = response_extractor(res) if res is not None else {}
                    _apply_llm_attrs(span, req=req, res=res_info)
                    span.set_status(Status(StatusCode.OK))
                    return res
                except BaseException as e:
                    _apply_llm_attrs(span, req=req, res=None)
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        return wrapper

    def _decorator(fn):
        return _decorate_async(fn) if asyncio.iscoroutinefunction(fn) else _decorate_sync(fn)

    return _decorator
