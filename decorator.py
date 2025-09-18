# oitrace/decorator.py
from __future__ import annotations
import asyncio, inspect, json, math, time
from functools import wraps
from typing import Any, Callable, Dict, Tuple

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

# OpenInference semantic conventions (generic attrs)
# See: openinference.semantic_conventions.attributes
try:
    from openinference.semconv.trace import SpanAttributes
except Exception:
    # fall back names if package layout changes; keep strings to avoid import brittleness
    class SpanAttributes:
        INPUT_VALUE = "openinference.input.value"
        INPUT_MIME_TYPE = "openinference.input.mime_type"
        OUTPUT_VALUE = "openinference.output.value"
        OUTPUT_MIME_TYPE = "openinference.output.mime_type"
        ERROR_MESSAGE = "openinference.error.message"
        ERROR_TYPE = "openinference.error.type"
        ERROR_STACKTRACE = "openinference.error.stacktrace"
        # generic keys for user metadata
        APP_COMPONENT = "openinference.app.component"
        APP_OPERATION = "openinference.app.operation"


Jsonable = Any
Redactor = Callable[[str, Any], bool]
Enricher = Callable[[Dict[str, Any]], Dict[str, Any]]


def _shorten(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    head = math.floor(limit * 0.7)
    tail = limit - head - 3
    return f"{s[:head]}...{s[-tail:]}"


def _jsonable(obj: Any, *, max_string: int, max_collection: int) -> Jsonable:
    """Best-effort, safe serialization that won’t explode on big/odd objects."""
    # primitives
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    # strings
    if isinstance(obj, str):
        return _shorten(obj, max_string)
    # bytes
    if isinstance(obj, (bytes, bytearray)):
        return f"<bytes:{len(obj)}>"
    # simple mappings
    if isinstance(obj, dict):
        out = {}
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_collection:
                out["..."] = f"+{len(obj)-max_collection} more"
                break
            key = str(k)
            out[key] = _jsonable(v, max_string=max_string, max_collection=max_collection)
        return out
    # simple sequences
    if isinstance(obj, (list, tuple, set)):
        seq = list(obj)
        trimmed = seq[:max_collection]
        res = [
            _jsonable(x, max_string=max_string, max_collection=max_collection)
            for x in trimmed
        ]
        if len(seq) > max_collection:
            res.append(f"... (+{len(seq)-max_collection} more)")
        return res if not isinstance(obj, tuple) else tuple(res)

    # dataclasses / pydantic / attrs – best effort via __dict__:
    for attr_name in ("model_dump", "dict"):  # pydantic v2 / v1
        if hasattr(obj, attr_name):
            try:
                d = getattr(obj, attr_name)()
                return _jsonable(d, max_string=max_string, max_collection=max_collection)
            except Exception:
                pass

    if hasattr(obj, "__dict__"):
        try:
            return _jsonable(vars(obj), max_string=max_string, max_collection=max_collection)
        except Exception:
            pass

    # final fallback
    return _shorten(repr(obj), max_string)


def _filter_args_kwargs(
    fn: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Bind args/kwargs to parameter names for cleaner display."""
    try:
        sig = inspect.signature(fn)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except Exception:
        # if binding fails, return positional + kwargs
        out = {f"arg{i}": v for i, v in enumerate(args)}
        out.update(kwargs)
        return out


def trace_io(
    *,
    name: str | None = None,
    tracer: trace.Tracer | None = None,
    component: str | None = None,
    operation: str | None = None,
    capture_inputs: bool = True,
    capture_outputs: bool = True,
    input_mime_type: str = "application/json",
    output_mime_type: str = "application/json",
    redact: Redactor | None = None,
    max_string: int = 4_000,
    max_collection: int = 100,
    enrich: Enricher | None = None,
):
    """
    Decorate any function and emit an OpenInference span with inputs/outputs:
      - name: override span name (defaults to fn.__qualname__)
      - component/operation: freeform tags like 'fastapi.handler' / 'predict'
      - redact(key, value)->bool: return True to omit/summarize a field
      - enrich(meta)->meta: add/modify attributes before they’re set on the span

    Works for sync and async functions.
    """

    tracer = tracer or trace.get_tracer(__name__)

    def _make_attrs_from_io(fn, args, kwargs, result=None, error: BaseException | None = None):
        attrs: Dict[str, Any] = {}
        # app metadata
        if component:
            attrs[SpanAttributes.APP_COMPONENT] = component
        if operation:
            attrs[SpanAttributes.APP_OPERATION] = operation

        # inputs
        if capture_inputs:
            bound = _filter_args_kwargs(fn, args, kwargs)
            inputs_sanitized = {}
            for k, v in bound.items():
                if redact and redact(k, v):
                    inputs_sanitized[k] = "<redacted>"
                else:
                    inputs_sanitized[k] = _jsonable(v, max_string=max_string, max_collection=max_collection)
            attrs[SpanAttributes.INPUT_VALUE] = json.dumps(inputs_sanitized, ensure_ascii=False)
            attrs[SpanAttributes.INPUT_MIME_TYPE] = input_mime_type

        # outputs or error
        if error is None and capture_outputs:
            try:
                out_val = _jsonable(result, max_string=max_string, max_collection=max_collection)
                attrs[SpanAttributes.OUTPUT_VALUE] = json.dumps(out_val, ensure_ascii=False)
                attrs[SpanAttributes.OUTPUT_MIME_TYPE] = output_mime_type
            except Exception as ser_err:
                attrs[SpanAttributes.OUTPUT_VALUE] = f"<unserializable: {ser_err!r}>"
        elif error is not None:
            attrs[SpanAttributes.ERROR_TYPE] = type(error).__name__
            attrs[SpanAttributes.ERROR_MESSAGE] = str(error)

        # user enrichment hook
        if enrich:
            try:
                attrs = enrich(attrs) or attrs
            except Exception:
                pass

        return attrs

    def _decorate_sync(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            span_name = name or fn.__qualname__
            start = time.perf_counter()
            with tracer.start_as_current_span(span_name, kind=SpanKind.INTERNAL) as span:
                try:
                    res = fn(*args, **kwargs)
                    attrs = _make_attrs_from_io(fn, args, kwargs, result=res, error=None)
                    for k, v in attrs.items():
                        span.set_attribute(k, v)
                    span.set_status(Status(StatusCode.OK))
                    return res
                except BaseException as e:
                    attrs = _make_attrs_from_io(fn, args, kwargs, result=None, error=e)
                    for k, v in attrs.items():
                        span.set_attribute(k, v)
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
                finally:
                    span.add_event("duration.ms", {"value": int((time.perf_counter() - start) * 1000)})
        return wrapper

    def _decorate_async(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            span_name = name or fn.__qualname__
            start = time.perf_counter()
            with tracer.start_as_current_span(span_name, kind=SpanKind.INTERNAL) as span:
                try:
                    res = await fn(*args, **kwargs)
                    attrs = _make_attrs_from_io(fn, args, kwargs, result=res, error=None)
                    for k, v in attrs.items():
                        span.set_attribute(k, v)
                    span.set_status(Status(StatusCode.OK))
                    return res
                except BaseException as e:
                    attrs = _make_attrs_from_io(fn, args, kwargs, result=None, error=e)
                    for k, v in attrs.items():
                        span.set_attribute(k, v)
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
                finally:
                    span.add_event("duration.ms", {"value": int((time.perf_counter() - start) * 1000)})
        return wrapper

    def _decorator(fn: Callable[..., Any]):
        if asyncio.iscoroutinefunction(fn):
            return _decorate_async(fn)
        return _decorate_sync(fn)

    return _decorator
