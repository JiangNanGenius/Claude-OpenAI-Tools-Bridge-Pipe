"""OpenWebUI Pipeline: Claude (Anthropic Messages API) ⇄ OpenAI ChatCompletions

Name: Claude OpenAI Tools Bridge Pipe
Author: JiangNanGenius
Repo: https://github.com/JiangNanGenius/

English
- Exposes Anthropic Claude models in OpenWebUI using OpenAI ChatCompletions format.
- Native tool UI: converts Anthropic tool_use/tool_result ↔ OpenAI tool_calls/tool messages.
- Claude Extended Thinking + tools:
  - Replays signed thinking blocks (thinking + signature) before tool_use on tool-continuation turns.
  - Fixes the common error: "Expected `thinking` or `redacted_thinking`, but found `tool_use`".
- Thinking display (hard-coded for stability):
  - Streams Claude thinking via `delta.reasoning_content` (no <think> tag hacks).
  - On tool-call turns, any normal assistant text is also routed to `reasoning_content`,
    so only the final answer is shown as normal content (prevents "analysis leaking" into正文 and
    avoids collapsed blocks "popping" into main text).

中文
- 在 OpenWebUI 中以 OpenAI ChatCompletions 兼容格式接入 Anthropic Claude。
- 原生工具 UI：Anthropic 的 tool_use/tool_result ↔ OpenAI 的 tool_calls/tool 消息互转。
- Extended Thinking + 工具调用：
  - 缓存并回放带 signature 的 thinking block，保证工具续轮时 tool_use 前面一定是 thinking。
  - 修复常见报错：Expected thinking/redacted_thinking, but found tool_use。
- 思考显示（已写死，减少折叠/串流异常）：
  - 使用 OpenWebUI 支持的 `delta.reasoning_content` 流式字段展示思考（不再输出 <think> 标签）。
  - 在“会调用工具”的回合，把模型输出的普通文本也强制归入 reasoning_content，
    这样工具回合不会在正文里混出“步骤/分析”，也更不容易出现折叠内容突然崩出来的问题。
"""


__version__ = "6.3.1"
__author__ = "JiangNanGenius"
__repo__ = "https://github.com/JiangNanGenius/"

import os
import json
import re
import time
import uuid
import logging
import threading

# Optional token counting fallback
try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore

import sys
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, Field, field_validator
from open_webui.utils.misc import pop_system_message


def _env_bool(key: str, default: bool) -> bool:
    """Parse common truthy/falsey env var strings into bool."""
    v = os.getenv(key, None)
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off", ""):
        return False
    # Fallback: non-empty string treated as True
    return True


# Set up logging
logger = logging.getLogger(__name__)


class AnthropicAPIError(Exception):
    """Custom exception for Anthropic API errors."""

    def __init__(self, status_code: int, message: str):
        super().__init__(f"Anthropic API Error ({status_code}): {message}")
        self.status_code = status_code
        self.message = message


class ImageSourceType(Enum):
    """Supported image source types."""

    BASE64 = "base64"
    URL = "url"


class ContentType(Enum):
    """Supported content types."""

    TEXT = "text"
    IMAGE = "image"
    THINKING = "thinking"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


class EventType(Enum):
    """Event types for streaming responses."""

    MESSAGE_START = "message_start"
    CONTENT_BLOCK_START = "content_block_start"
    CONTENT_BLOCK_DELTA = "content_block_delta"
    CONTENT_BLOCK_STOP = "content_block_stop"
    MESSAGE_DELTA = "message_delta"
    MESSAGE = "message"  # some proxies send a full message event
    MESSAGE_STOP = "message_stop"
    ERROR = "error"
    PING = "ping"


class DeltaType(Enum):
    """Delta types in streaming responses."""

    TEXT_DELTA = "text_delta"
    THINKING_DELTA = "thinking_delta"
    INPUT_JSON_DELTA = "input_json_delta"
    SIGNATURE_DELTA = "signature_delta"


class ThinkingState(Enum):
    """State of thinking in streaming responses."""

    NOT_STARTED = -1
    IN_PROGRESS = 0
    COMPLETED = 1


@dataclass
class ModelConfig:
    """Configuration for a specific Anthropic model."""

    id: str
    name: str
    api_identifier: str = field(default="")

    def __post_init__(self):
        if not self.api_identifier:
            self.api_identifier = self.id


class Pipe:
    """
    OpenWebUI Pipeline for Anthropic Messages API.

    Adds native tool calling support by translating:
      - OpenAI-style tools/tool_choice + tool_calls/tool messages (OpenWebUI internal format)
      - to Anthropic tools/tool_choice + tool_use/tool_result content blocks

    And translates Anthropic tool_use back to OpenAI-style tool_calls so OpenWebUI can run tools.
    """

    class Valves(BaseModel):
        # --- Auth / Endpoint ---
        ANTHROPIC_API_KEY: str = Field(
            default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
        )
        # Base URL for Anthropic Messages API. Examples:
        # - https://api.anthropic.com
        # - https://api.openai-proxy.org/anthropic
        # If you provide a full Messages endpoint (ending with /v1/messages), it will be used as-is.
        BASE_URL: str = Field(
            default_factory=lambda: os.getenv(
                "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
            )
        )

        # Timeouts (requests timeout=(connect, read))
        CONNECT_TIMEOUT: float = Field(
            default_factory=lambda: float(os.getenv("ANTHROPIC_CONNECT_TIMEOUT", "10"))
        )
        READ_TIMEOUT: float = Field(
            default_factory=lambda: float(os.getenv("ANTHROPIC_READ_TIMEOUT", "300"))
        )

        # Prefer mapped/fallback models before the requested model (useful for proxies that reject Claude 4.x)
        PREFER_MAPPED_MODEL_FIRST: bool = Field(
            default_factory=lambda: _env_bool(
                "ANTHROPIC_PREFER_MAPPED_MODEL_FIRST", True
            )
        )

        # --- Logging ---
        LOG_ENABLED: bool = Field(
            default_factory=lambda: _env_bool("ANTHROPIC_LOG_ENABLED", False)
        )
        LOG_TO_CONSOLE: bool = Field(
            default_factory=lambda: _env_bool("ANTHROPIC_LOG_TO_CONSOLE", True)
        )
        LOG_LEVEL: str = Field(
            default_factory=lambda: os.getenv("ANTHROPIC_LOG_LEVEL", "WARNING")
        )

        # Tool call streaming behavior:
        # - "single": emit each tool_call once with full arguments (avoids some OpenWebUI delta-merge bugs)
        # - "delta":  emit tool_call arguments incrementally as received from Anthropic stream
        TOOL_CALL_STREAM_MODE: str = Field(
            default_factory=lambda: os.getenv(
                "ANTHROPIC_TOOL_CALL_STREAM_MODE", "single"
            )
        )

        # If True, allow Anthropic "extended thinking" even when tool calling is enabled.
        # This requires preserving the signed thinking block (including `signature`) across the tool loop.
        ALLOW_THINKING_WITH_TOOLS: bool = Field(
            default_factory=lambda: _env_bool(
                "ANTHROPIC_ALLOW_THINKING_WITH_TOOLS", True
            )
        )

        # If True, buffer assistant text while streaming; if the model uses tools, drop buffered text so the
        # assistant emits ONLY tool_calls in that turn (avoids some OpenWebUI delta-merge bugs).
        SUPPRESS_CONTENT_ON_TOOL_CALLS: bool = Field(
            default_factory=lambda: _env_bool(
                "ANTHROPIC_SUPPRESS_CONTENT_ON_TOOL_CALLS", True
            )
        )

        # If True AND SUPPRESS_CONTENT_ON_TOOL_CALLS is enabled, flush any buffered assistant text immediately
        # BEFORE the first tool_call is emitted.
        #
        # Why this exists:
        # - With SUPPRESS_CONTENT_ON_TOOL_CALLS, we buffer normal assistant text and (by default) drop it once a tool is used.
        # - That improves compatibility with some OpenWebUI builds, but it also means you won't see any model text that
        #   appears before a tool call (e.g. "I'll search X then scrape Y").
        # - Enabling this valve lets you *see* that pre-tool text while still keeping the buffering behavior for the
        #   remainder of the tool-calling turn.
        FLUSH_BUFFERED_TEXT_BEFORE_TOOL_CALLS: bool = Field(
            default_factory=lambda: _env_bool(
                "ANTHROPIC_FLUSH_BUFFERED_TEXT_BEFORE_TOOL_CALLS", False
            )
        )

        # Signed thinking blocks are cached by tool_call_id so we can replay them on the subsequent tool_result request.
        THINKING_SIG_CACHE_MAX_ITEMS: int = Field(
            default_factory=lambda: int(
                os.getenv("ANTHROPIC_THINKING_SIG_CACHE_MAX_ITEMS", "2048")
            )
        )
        THINKING_SIG_CACHE_TTL_SEC: int = Field(
            default_factory=lambda: int(
                os.getenv("ANTHROPIC_THINKING_SIG_CACHE_TTL_SEC", "3600")
            )
        )

        # --- Usage / Token reporting ---
        # If True, include OpenAI-style `usage` in OpenAI-formatted responses.
        RETURN_TOKENS: bool = Field(
            default_factory=lambda: _env_bool("ANTHROPIC_RETURN_TOKENS", False)
        )
        # If the upstream API does not return token usage, estimate tokens with tiktoken (approximate).
        FALLBACK_TIKTOKEN: bool = Field(
            default_factory=lambda: _env_bool("ANTHROPIC_FALLBACK_TIKTOKEN", True)
        )
        # tiktoken encoding to use for fallback estimates.
        TIKTOKEN_ENCODING: str = Field(
            default_factory=lambda: os.getenv(
                "ANTHROPIC_TIKTOKEN_ENCODING", "cl100k_base"
            )
        )

        # --- Model compatibility / fallback ---
        # Some Anthropic-compatible proxies support only a subset of models.
        # If an upstream call fails with an "unsupported model" error, the Pipe can retry with fallbacks.
        ENABLE_MODEL_FALLBACK: bool = Field(
            default_factory=lambda: _env_bool("ANTHROPIC_ENABLE_MODEL_FALLBACK", False)
        )
        # Comma-separated list of fallback model names to try (in order).
        # Example: "claude-3-7-sonnet-latest,claude-3-5-sonnet-latest,claude-3-5-haiku-latest"
        MODEL_FALLBACK_CHAIN: str = Field(
            default_factory=lambda: os.getenv(
                "ANTHROPIC_MODEL_FALLBACK_CHAIN",
                "claude-3-7-sonnet-latest,claude-3-5-sonnet-latest,claude-3-5-haiku-latest",
            )
        )
        # Optional explicit mappings (comma-separated "from=to").
        # Example: "claude-4-5-sonnet-latest=claude-3-7-sonnet-latest,claude-4-sonnet-latest=claude-3-7-sonnet-latest"
        MODEL_MAP: str = Field(
            default_factory=lambda: os.getenv("ANTHROPIC_MODEL_MAP", "")
        )
        # If enabled, apply a conservative built-in downgrade map (e.g. claude-4-5 → claude-3-7).
        # Useful only for providers/proxies that do NOT support the newer model ids.
        ENABLE_BUILTIN_MODEL_DOWNGRADE: bool = Field(
            default=False,
            description="Enable built-in model downgrade mapping for incompatible providers (OFF recommended if your proxy supports the requested model ids).",
        )

        # If enabled, use ModelConfig.api_identifier when calling the upstream Anthropic endpoint.
        # OpenWebUI will still see the original requested model id in responses.
        USE_API_IDENTIFIER: bool = Field(
            default=True,
            description="Call upstream using each model's api_identifier instead of its id (optional).",
        )

        @field_validator("ANTHROPIC_API_KEY")
        def check_api_key(cls, value: str) -> str:
            if not value:
                logger.warning("ANTHROPIC_API_KEY is not set")
            return value

        @field_validator("BASE_URL")
        def check_base_url(cls, value: str) -> str:
            if not value:
                logger.warning(
                    "BASE_URL is empty; falling back to default proxy base URL"
                )
                return "https://api.openai-proxy.org/anthropic"
            v = value.strip()
            # Allow user to pass full endpoint; otherwise treat as base URL.
            parsed = urlparse(v if "://" in v else f"https://{v}")
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                logger.warning(
                    f"Invalid BASE_URL: {value}. Using default proxy base URL instead."
                )
                return "https://api.openai-proxy.org/anthropic"
            return v

        @field_validator("LOG_LEVEL")
        def check_log_level(cls, value: str) -> str:
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if value.upper() not in valid_levels:
                logger.warning(f"Invalid log level: {value}. Using WARNING instead.")
                return "WARNING"
            return value.upper()

        @field_validator("TOOL_CALL_STREAM_MODE")
        def check_tool_call_stream_mode(cls, value: str) -> str:
            v = (value or "").strip().lower()
            if v not in ("single", "delta"):
                logger.warning(
                    f"Invalid TOOL_CALL_STREAM_MODE: {value}. Using 'single'."
                )
                return "single"
            return v

    API_VERSION = "2023-06-01"
    API_ENDPOINT = "https://api.openai-proxy.org/anthropic/v1/messages"
    MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB per image
    MAX_TOTAL_IMAGE_SIZE_BYTES = 100 * 1024 * 1024  # 100MB total
    DEFAULT_TIMEOUT = (3.05, 60)  # (connect timeout, read timeout)

    @staticmethod
    def _resolve_messages_endpoint(base_url: str) -> str:
        """Resolve BASE_URL to the full Messages API endpoint."""
        b = (base_url or "").strip().rstrip("/")
        if not b:
            b = "https://api.openai-proxy.org/anthropic"
        # If user passed the full endpoint, keep it.
        if b.endswith("/v1/messages"):
            return b
        return f"{b}/v1/messages"

    def _configure_logging(self) -> None:
        """Configure console logging according to valves."""
        # Always set level (even if disabled) so future enabling is consistent.
        level = getattr(logging, self.valves.LOG_LEVEL, logging.WARNING)

        if not self.valves.LOG_ENABLED:
            # Disable this logger entirely (no console noise).
            logger.disabled = True
            return

        logger.disabled = False
        logger.setLevel(level)

        # Attach a StreamHandler only if requested AND not already attached.
        if self.valves.LOG_TO_CONSOLE:
            has_stream = any(
                isinstance(h, logging.StreamHandler) for h in logger.handlers
            )
            if not has_stream:
                handler = logging.StreamHandler(stream=sys.stdout)
                handler.setLevel(level)
                handler.setFormatter(
                    logging.Formatter(
                        "[%(asctime)s] [%(levelname)s] [claude-openai-tools] %(message)s"
                    )
                )
                logger.addHandler(handler)

        # Avoid double logging via root handlers unless the user wants that.
        logger.propagate = False

    def _log(self, level: str, msg: str) -> None:
        """Lightweight internal logging helper.

        In OpenWebUI containers, python logging handlers may not always surface in the console.
        So when LOG_ENABLED is on, we also emit a `print()` line (flush=True) for visibility.
        """
        try:
            valves = getattr(self, "valves", None)
            if (
                valves
                and getattr(valves, "LOG_ENABLED", False)
                and getattr(valves, "LOG_TO_CONSOLE", True)
            ):
                print(f"[claude-openai-tools] {level.upper()}: {msg}", flush=True)
        except Exception:
            pass

        if logger.disabled:
            return
        getattr(logger, level, logger.info)(msg)

    def _prune_thinking_sig_cache(self) -> None:
        """Prune expired/oversized signed-thinking cache (best-effort)."""
        ttl = int(getattr(self.valves, "THINKING_SIG_CACHE_TTL_SEC", 3600) or 3600)
        max_items = int(
            getattr(self.valves, "THINKING_SIG_CACHE_MAX_ITEMS", 2048) or 2048
        )
        now = time.time()
        with self._thinking_sig_cache_lock:
            # Drop expired
            expired = [
                k
                for k, v in self._thinking_sig_cache.items()
                if (now - float(v.get("ts", now))) > ttl
            ]
            for k in expired:
                self._thinking_sig_cache.pop(k, None)
            # Drop oldest if oversized
            if max_items > 0 and len(self._thinking_sig_cache) > max_items:
                items = sorted(
                    self._thinking_sig_cache.items(),
                    key=lambda kv: float(kv[1].get("ts", 0.0)),
                )
                for k, _ in items[: max(0, len(items) - max_items)]:
                    self._thinking_sig_cache.pop(k, None)

    def _cache_signed_thinking(
        self, tool_ids: List[str], thinking: str, signature: str
    ) -> None:
        """Cache a signed thinking block for one or more tool_call_ids."""
        if not tool_ids or not thinking or not signature:
            return
        self._prune_thinking_sig_cache()
        now = time.time()
        with self._thinking_sig_cache_lock:
            for tid in tool_ids:
                if not tid:
                    continue
                self._thinking_sig_cache[tid] = {
                    "thinking": thinking,
                    "signature": signature,
                    "ts": now,
                }

    def _get_signed_thinking(self, tool_id: str) -> Optional[Dict[str, str]]:
        """Return cached signed thinking for a tool_call_id if available."""
        if not tool_id:
            return None
        self._prune_thinking_sig_cache()
        with self._thinking_sig_cache_lock:
            v = self._thinking_sig_cache.get(tool_id)
            if not isinstance(v, dict):
                return None
            thinking = v.get("thinking")
            signature = v.get("signature")
            if (
                isinstance(thinking, str)
                and isinstance(signature, str)
                and thinking
                and signature
            ):
                return {"thinking": thinking, "signature": signature}
            return None

    def __init__(self):
        self.type = "manifold"
        self.id = "claude_openai_tools"
        self.name = "claude-openai-tools/"

        # NOTE:
        # OpenWebUI will override `self.valves` from the UI at runtime.
        # So we only set env-backed defaults here, and we re-apply logging/endpoint on every request.
        self.valves = self.Valves()
        self.api_endpoint = self._resolve_messages_endpoint(self.valves.BASE_URL)

        # Configure logging once for startup (will be reconfigured per-request as well).
        self._configure_logging()
        self._log(
            "info", f"Using Anthropic Messages endpoint (init): {self.api_endpoint}"
        )

        self._models = self._initialize_models()

        # Cache for Anthropic signed thinking blocks when using extended thinking + tool calling.
        # Keyed by tool_call_id (Anthropic tool_use.id). This is in-memory and per-process.
        self._thinking_sig_cache: Dict[str, Dict[str, Any]] = {}
        self._thinking_sig_cache_lock = threading.Lock()

    def _initialize_models(self) -> List[ModelConfig]:
        """Initialize available model configurations."""
        return [
            # 3.x family (widely supported by many Anthropic-compatible proxies)
            ModelConfig(
                id="claude-3-7-sonnet-latest",
                name="claude-3-7-sonnet-latest",
                api_identifier="claude-3-7-sonnet-latest",
            ),
            ModelConfig(
                id="claude-3-7-sonnet-latest-extended-thinking",
                name="claude-3-7-sonnet-latest (extended thinking)",
                api_identifier="claude-3-7-sonnet-latest",
            ),
            ModelConfig(
                id="claude-3-5-sonnet-latest",
                name="claude-3-5-sonnet-latest",
                api_identifier="claude-3-5-sonnet-latest",
            ),
            ModelConfig(
                id="claude-3-5-sonnet-latest-extended-thinking",
                name="claude-3-5-sonnet-latest (extended thinking)",
                api_identifier="claude-3-5-sonnet-latest",
            ),
            ModelConfig(
                id="claude-3-5-haiku-latest",
                name="claude-3-5-haiku-latest",
                api_identifier="claude-3-5-haiku-latest",
            ),
            ModelConfig(
                id="claude-3-5-haiku-latest-extended-thinking",
                name="claude-3-5-haiku-latest (extended thinking)",
                api_identifier="claude-3-5-haiku-latest",
            ),
            ModelConfig(
                id="claude-3-opus-latest",
                name="claude-3-opus-latest",
                api_identifier="claude-3-opus-latest",
            ),
            ModelConfig(
                id="claude-3-opus-latest-extended-thinking",
                name="claude-3-opus-latest (extended thinking)",
                api_identifier="claude-3-opus-latest",
            ),
            ModelConfig(
                id="claude-4-sonnet-latest",
                name="claude-4-sonnet-latest",
                api_identifier="claude-sonnet-4-0",
            ),
            ModelConfig(
                id="claude-4-sonnet-latest-extended-thinking",
                name="claude-4-sonnet-latest (extended thinking)",
                api_identifier="claude-sonnet-4-0",
            ),
            ModelConfig(
                id="claude-4-opus-latest",
                name="claude-4-opus-latest",
                api_identifier="claude-opus-4-0",
            ),
            ModelConfig(
                id="claude-4-opus-latest-extended-thinking",
                name="claude-4-opus-latest (extended thinking)",
                api_identifier="claude-opus-4-0",
            ),
            ModelConfig(
                id="claude-4-5-haiku-latest",
                name="claude-4-5-haiku-latest",
                api_identifier="claude-haiku-4-5",
            ),
            ModelConfig(
                id="claude-4-5-haiku-latest-extended-thinking",
                name="claude-4-5-haiku-latest (extended thinking)",
                api_identifier="claude-haiku-4-5",
            ),
            ModelConfig(
                id="claude-4-5-sonnet-latest",
                name="claude-4-5-sonnet-latest",
                api_identifier="claude-sonnet-4-5",
            ),
            ModelConfig(
                id="claude-4-5-sonnet-latest-extended-thinking",
                name="claude-4-5-sonnet-latest (extended thinking)",
                api_identifier="claude-sonnet-4-5",
            ),
            ModelConfig(
                id="claude-opus-4-5-latest",
                name="claude-opus-4-5-latest",
                api_identifier="claude-opus-4-5",
            ),
            ModelConfig(
                id="claude-opus-4-5-latest-extended-thinking",
                name="claude-opus-4-5-latest (extended thinking)",
                api_identifier="claude-opus-4-5",
            ),
        ]

    @lru_cache(maxsize=1)
    def get_anthropic_models(self) -> List[Dict[str, str]]:
        """Return a list of available Anthropic models."""
        return [{"id": model.id, "name": model.name} for model in self._models]

    def pipes(self) -> List[Dict[str, str]]:
        """Alias for get_anthropic_models."""
        return self.get_anthropic_models()

    # ---------------------------
    # Utilities: images
    # ---------------------------

    def process_image(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an image for inclusion in a message. Handles both base64 and URL images.
        OpenWebUI sends OpenAI-style: {"type":"image_url","image_url":{"url":"..."}}
        """
        image_url = image_data["image_url"]["url"]

        # Process image via base64 data URL
        if image_url.startswith("data:image"):
            try:
                header, base64_data = image_url.split(",", 1)
                media_type = header.split(";")[0].split(":")[1]
            except Exception as e:
                raise ValueError(f"Invalid base64 image data: {e}")

            image_size = len(base64_data) * 3 / 4
            if image_size > self.MAX_IMAGE_SIZE_BYTES:
                raise ValueError(
                    f"Image size {image_size / (1024 * 1024):.2f}MB exceeds {self.MAX_IMAGE_SIZE_BYTES / (1024 * 1024):.2f}MB limit"
                )

            return {
                "type": ContentType.IMAGE.value,
                "source": {
                    "type": ImageSourceType.BASE64.value,
                    "media_type": media_type,
                    "data": base64_data,
                },
            }

        # Process image via URL
        parsed_url = urlparse(image_url)
        if not (parsed_url.scheme and parsed_url.netloc):
            raise ValueError(f"Invalid image URL: {image_url}")

        # HEAD to estimate size (best-effort)
        response = requests.head(
            image_url,
            allow_redirects=True,
            timeout=(
                getattr(self.valves, "CONNECT_TIMEOUT", 10),
                getattr(self.valves, "READ_TIMEOUT", 300),
            ),
        )
        response.raise_for_status()
        content_length = int(response.headers.get("content-length", 0))
        if content_length and content_length > self.MAX_IMAGE_SIZE_BYTES:
            raise ValueError(
                f"Image size {content_length / (1024 * 1024):.2f}MB exceeds {self.MAX_IMAGE_SIZE_BYTES / (1024 * 1024):.2f}MB limit"
            )

        return {
            "type": ContentType.IMAGE.value,
            "source": {
                "type": ImageSourceType.URL.value,
                "url": image_url,
            },
        }

    # ---------------------------
    # Utilities: tools + messages
    # ---------------------------

    @staticmethod
    def _safe_json_loads(s: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(s, str):
            return None
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
            return {"_value": obj}
        except Exception:
            return None


    @staticmethod
    def _strip_reasoning_tags(text: str) -> str:
        """Remove OpenWebUI-style reasoning tags from plain text.

        OpenWebUI may serialize reasoning during a native tool loop as:
            <think>...</think>
        and include that string in assistant messages sent back to the provider.

        For Anthropic *extended thinking*, we must replay the signed thinking blocks
        instead; sending the serialized <think>...</think> back to Claude can cause
        duplication and can also confuse the model.

        This helper removes the tag blocks (including their inner content).
        """
        if not text:
            return text

        # Remove the most common reasoning wrappers.
        patterns = [
            r"<think>.*?</think>",
            r"<thinking>.*?</thinking>",
            r"<reasoning>.*?</reasoning>",
            r"<reason>.*?</reason>",
            # OpenWebUI may export reasoning blocks as <details type="reasoning">...</details>
            r'<details\s+[^>]*type="reasoning"[^>]*>.*?</details>',
        ]
        out = text
        for p in patterns:
            out = re.sub(p, "", out, flags=re.IGNORECASE | re.DOTALL)
        return out


    @staticmethod
    def _openai_tools_to_anthropic(tools: Any) -> Optional[List[Dict[str, Any]]]:
        """
        OpenAI tools: [{"type":"function","function":{"name","description","parameters"}}]
        Anthropic tools: [{"name","description","input_schema"}]
        """
        if not tools:
            return None
        if not isinstance(tools, list):
            return None

        out: List[Dict[str, Any]] = []
        for t in tools:
            if not isinstance(t, dict):
                continue

            # OpenAI tool schema
            if t.get("type") == "function" and isinstance(t.get("function"), dict):
                fn = t["function"]
                name = fn.get("name")
                if not name:
                    continue
                out.append(
                    {
                        "name": name,
                        "description": fn.get("description", "") or "",
                        "input_schema": fn.get("parameters")
                        or {"type": "object", "properties": {}},
                    }
                )
                continue

            # Legacy OpenAI "functions" style sometimes passed as tool objects
            if "name" in t and ("parameters" in t or "input_schema" in t):
                out.append(
                    {
                        "name": t.get("name"),
                        "description": t.get("description", "") or "",
                        "input_schema": t.get("input_schema")
                        or t.get("parameters")
                        or {"type": "object", "properties": {}},
                    }
                )
                continue

        # Sanitize schemas for Anthropic: input_schema must be a JSON Schema object (dict) and should be an object schema.
        for _t in out:
            schema = _t.get("input_schema")
            if not isinstance(schema, dict):
                schema = {"type": "object", "properties": {}}
            if schema.get("type") is None:
                schema["type"] = "object"
            if schema.get("type") == "object" and "properties" not in schema:
                schema["properties"] = {}
            _t["input_schema"] = schema
            # Help models avoid translating the tool name by repeating it in the description.
            try:
                n = str(_t.get("name") or "").strip()
                if n:
                    d = str(_t.get("description") or "").strip()
                    prefix = f"Tool name (identifier): {n}. "
                    if not d.startswith(prefix):
                        _t["description"] = prefix + d
            except Exception:
                pass

        return out or None

    @staticmethod
    def _best_effort_tool_name(raw_name: Any, valid_names: Optional[set] = None) -> str:
        """Best-effort normalization for tool names.

        Why this exists:
        - Some clients concatenate tool name deltas incorrectly (e.g., 'foofoofoo').
        - Some models may copy corrupted names from history.
        - Anthropic validates tool_use.name length (<=200 chars).

        Strategy:
        1) Exact match against valid tool names (if provided).
        2) Detect repetitions of a valid name (with/without whitespace).
        3) Substring match against valid tool names.
        4) Generic repetition collapse (smallest repeating unit).
        5) Clamp to 200 chars as a final safeguard.
        """
        name = str(raw_name or "").strip()
        if not name:
            return ""

        if valid_names and name in valid_names:
            return name

        # Remove whitespace for robustness
        name_nospace = re.sub(r"\s+", "", name)

        if valid_names:
            # Prefer longer valid names to avoid picking a short substring.
            for vn in sorted([v for v in valid_names if v], key=len, reverse=True):
                # exact or whitespace-stripped exact
                if name == vn or name_nospace == vn:
                    return vn
                # collapse repetitions like 'foofoofoo' -> 'foo'
                if len(name) > len(vn) and name.replace(vn, "") == "":
                    return vn
                if len(name_nospace) > len(vn) and name_nospace.replace(vn, "") == "":
                    return vn

            # Substring match for concatenations (pick the longest match)
            best = ""
            for vn in valid_names:
                if vn and vn in name_nospace and len(vn) > len(best):
                    best = vn
            if best:
                return best

        # Generic repetition collapse (no tool list / unknown tool names)
        cand = name_nospace
        # Bound the search so it stays cheap even if cand is huge.
        max_unit = min(64, max(1, len(cand) // 2))
        for i in range(1, max_unit + 1):
            if len(cand) % i != 0:
                continue
            unit = cand[:i]
            if unit * (len(cand) // i) == cand:
                if re.match(r"^[A-Za-z0-9_.\-]{2,}$", unit):
                    return unit[:200]

        # Last resort: clamp length so Anthropic accepts the request.
        return name[:200]

    @staticmethod
    def _openai_tool_choice_to_anthropic(tool_choice: Any) -> Optional[Dict[str, Any]]:
        """
        OpenAI tool_choice can be:
          - "none" | "auto" | "required"
          - {"type":"function","function":{"name":"x"}}
        Anthropic tool_choice is typically:
          - {"type":"auto"} | {"type":"none"} | {"type":"any"} | {"type":"tool","name":"x"}
        """
        if tool_choice is None:
            return None

        if isinstance(tool_choice, str):
            tc = tool_choice.lower().strip()
            if tc == "none":
                return {"type": "none"}
            if tc == "auto":
                return {"type": "auto"}
            if tc == "required":
                return {"type": "any"}
            return None

        if isinstance(tool_choice, dict):
            # OpenAI forced function call
            if tool_choice.get("type") == "function" and isinstance(
                tool_choice.get("function"), dict
            ):
                name = tool_choice["function"].get("name")
                if name:
                    return {"type": "tool", "name": name}
            # Some callers may already pass anthropic-style
            if tool_choice.get("type") in ("auto", "none", "any"):
                return {"type": tool_choice.get("type")}
            if tool_choice.get("type") == "tool" and tool_choice.get("name"):
                return {"type": "tool", "name": tool_choice["name"]}

        return None

    def _convert_messages_for_anthropic(
        self, messages: List[Dict[str, Any]], valid_tool_names: Optional[set] = None
    ) -> Tuple[List[Dict[str, Any]], int, bool]:
        """
        Convert OpenWebUI/OpenAI-style messages into Anthropic Messages format.
        Returns: (processed_messages, total_image_size, saw_tool_messages)
        """
        processed_messages: List[Dict[str, Any]] = []
        total_image_size = 0
        saw_tools = False

        def add_content_blocks(dst: List[Dict[str, Any]], content: Any) -> None:
            nonlocal total_image_size
            if content is None:
                return

            # String content
            if isinstance(content, str):
                if content:
                    dst.append({"type": ContentType.TEXT.value, "text": content})
                return

            # OpenAI block list
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        # fallback
                        dst.append({"type": ContentType.TEXT.value, "text": str(item)})
                        continue
                    t = item.get("type")
                    if t == ContentType.TEXT.value:
                        dst.append(
                            {
                                "type": ContentType.TEXT.value,
                                "text": item.get("text", ""),
                            }
                        )
                    elif t == "image_url":
                        processed_image = self.process_image(item)
                        dst.append(processed_image)
                        if (
                            processed_image["source"]["type"]
                            == ImageSourceType.BASE64.value
                        ):
                            image_size = len(processed_image["source"]["data"]) * 3 / 4
                            total_image_size += image_size
                            if total_image_size > self.MAX_TOTAL_IMAGE_SIZE_BYTES:
                                raise ValueError(
                                    f"Total size of images exceeds {self.MAX_TOTAL_IMAGE_SIZE_BYTES / (1024 * 1024):.2f}MB limit"
                                )
                    elif t == ContentType.TOOL_RESULT.value:
                        # already anthropic tool_result block
                        dst.append(item)
                    elif t == ContentType.TOOL_USE.value:
                        # already anthropic tool_use block - sanitize name defensively
                        try:
                            if isinstance(item, dict):
                                _n = self._best_effort_tool_name(
                                    item.get("name"), valid_tool_names
                                )
                                if _n:
                                    item["name"] = _n[:200]
                        except Exception:
                            pass
                        dst.append(item)
                    elif t == ContentType.IMAGE.value:
                        # already anthropic image block
                        dst.append(item)
                    else:
                        # unknown block: keep as text
                        dst.append({"type": ContentType.TEXT.value, "text": str(item)})
                return

            # Any other type -> text
            dst.append({"type": ContentType.TEXT.value, "text": str(content)})

        for message in messages:
            role = message.get("role")
            content = message.get("content")

            # Tool result in OpenAI format
            if role == "tool":
                saw_tools = True
                tool_call_id = (
                    message.get("tool_call_id")
                    or message.get("tool_use_id")
                    or message.get("id")
                )
                if not tool_call_id:
                    tool_call_id = f"toolu_{uuid.uuid4().hex}"

                tool_content = content
                if tool_content is None:
                    tool_content = ""
                if not isinstance(tool_content, str):
                    try:
                        tool_content = json.dumps(tool_content, ensure_ascii=False)
                    except Exception:
                        tool_content = str(tool_content)

                self._log(
                    "debug",
                    f"TOOL_RESULT←OWUI tool_call_id={tool_call_id} content_len={len(tool_content)}",
                )
                processed_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": ContentType.TOOL_RESULT.value,
                                "tool_use_id": tool_call_id,
                                "content": tool_content,
                            }
                        ],
                    }
                )
                continue

            # Normal user/assistant
            # OpenWebUI may serialize reasoning into assistant content (e.g., <think>...</think>)
            # during native tool loops. Anthropic extended thinking requires signed thinking blocks,
            # so we strip these text tags here to avoid duplicating reasoning in the prompt.
            if role == "assistant" and isinstance(message.get("tool_calls"), list) and (message.get("tool_calls") or []):
                if isinstance(content, str):
                    content = self._strip_reasoning_tags(content)
                elif isinstance(content, list):
                    _clean: list = []
                    for _blk in content:
                        if (
                            isinstance(_blk, dict)
                            and _blk.get("type") == ContentType.TEXT.value
                            and isinstance(_blk.get("text"), str)
                        ):
                            _blk = dict(_blk)
                            _blk["text"] = self._strip_reasoning_tags(_blk.get("text") or "")
                        _clean.append(_blk)
                    content = _clean
            processed_content: List[Dict[str, Any]] = []
            add_content_blocks(processed_content, content)

            # Assistant tool_calls -> tool_use blocks
            if role == "assistant" and isinstance(message.get("tool_calls"), list):
                # Replay signed thinking (extended thinking + tools): OpenWebUI does not persist Anthropic
                # thinking signatures, so we cache them keyed by tool_call_id and re-inject here.
                if bool(getattr(self.valves, "ALLOW_THINKING_WITH_TOOLS", False)):
                    has_thinking = any(
                        isinstance(b, dict)
                        and b.get("type") == ContentType.THINKING.value
                        for b in (processed_content or [])
                    )
                    if not has_thinking:
                        cached = None
                        for tc0 in message.get("tool_calls") or []:
                            tid0 = tc0.get("id") if isinstance(tc0, dict) else None
                            if isinstance(tid0, str) and tid0:
                                cached = self._get_signed_thinking(tid0)
                                if cached:
                                    break
                        if cached:
                            processed_content.insert(
                                0,
                                {
                                    "type": ContentType.THINKING.value,
                                    "thinking": cached["thinking"],
                                    "signature": cached["signature"],
                                },
                            )
                saw_tools = True
                for tc in message["tool_calls"]:
                    if not isinstance(tc, dict):
                        continue
                    if tc.get("type") != "function":
                        continue
                    fn = tc.get("function") or {}
                    raw_name = fn.get("name")
                    name = self._best_effort_tool_name(raw_name, valid_tool_names)
                    if name:
                        name = name[:200]
                    if not name:
                        continue
                    args_str = fn.get("arguments", "") or ""
                    args_obj = self._safe_json_loads(args_str)
                    if args_obj is None:
                        # Keep raw arguments in a stable wrapper; Claude can still see it
                        args_obj = {"_raw_arguments": args_str}
                    processed_content.append(
                        {
                            "type": ContentType.TOOL_USE.value,
                            "id": tc.get("id") or f"toolu_{uuid.uuid4().hex}",
                            "name": name,
                            "input": args_obj,
                        }
                    )

            # If a user message already contains tool_result blocks, treat it as tool flow.
            if role in ("user", "assistant") and isinstance(processed_content, list):
                for b in processed_content:
                    if isinstance(b, dict) and b.get("type") in (
                        ContentType.TOOL_RESULT.value,
                        ContentType.TOOL_USE.value,
                    ):
                        saw_tools = True
                        break

            processed_messages.append({"role": role, "content": processed_content})

        return processed_messages, total_image_size, saw_tools

    # ---------------------------
    # Model selection + payload
    # ---------------------------

    @staticmethod
    def _parse_kv_csv(s: str) -> Dict[str, str]:
        """Parse comma-separated key=value pairs into a dict."""
        out: Dict[str, str] = {}
        if not s:
            return out
        for part in str(s).split(","):
            p = part.strip()
            if not p or "=" not in p:
                continue
            k, v = p.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k and v:
                out[k] = v
        return out

    def _to_api_identifier(self, model_short: str) -> str:
        """Translate an OpenWebUI model id to the upstream Anthropic 'api_identifier' if configured.

        - When USE_API_IDENTIFIER is OFF, this returns the input unchanged.
        - When ON, this tries to find a matching ModelConfig by id and returns its api_identifier.
          If no match is found, returns the input unchanged.

        This lets you keep OpenWebUI model ids stable while optionally calling the upstream with a
        different identifier (common when using proxies).
        """
        if not getattr(self.valves, "USE_API_IDENTIFIER", False):
            return model_short

        try:
            for m in getattr(self, "_models", []) or []:
                if getattr(m, "id", None) == model_short and getattr(
                    m, "api_identifier", None
                ):
                    return str(m.api_identifier)
        except Exception:
            # Never fail model selection due to identifier mapping.
            return model_short

        return model_short

    def _candidate_models_for_request(
        self, requested_model_id: str
    ) -> Tuple[List[str], bool]:
        """Return candidate upstream model names to try, and whether extended thinking was requested."""
        chosen, extended_thinking = self._select_model(requested_model_id)

        # Model map overrides (from=to)
        model_map = self._parse_kv_csv(getattr(self.valves, "MODEL_MAP", "") or "")

        # Fallback chain
        chain_raw = getattr(self.valves, "MODEL_FALLBACK_CHAIN", "") or ""
        chain = [m.strip() for m in chain_raw.split(",") if m.strip()]

        candidates: List[str] = []

        def _add(m: str) -> None:
            if m and m not in candidates:
                candidates.append(m)

        pref_map_first = bool(getattr(self.valves, "PREFER_MAPPED_MODEL_FIRST", True))

        if pref_map_first:
            # Try explicit/builtin mappings first (useful when the requested model is rejected by the proxy)
            if chosen in model_map:
                _add(model_map[chosen])
            rid = (requested_model_id or "").strip()
            if rid in model_map:
                _add(model_map[rid])
            if getattr(self.valves, "ENABLE_BUILTIN_MODEL_DOWNGRADE", False):
                builtin_map = {
                    "claude-4-5-sonnet-latest": "claude-3-7-sonnet-latest",
                    "claude-4-5-sonnet-latest-extended-thinking": "claude-3-7-sonnet-latest",
                    "claude-4-sonnet-latest": "claude-3-7-sonnet-latest",
                }
                if chosen in builtin_map:
                    _add(builtin_map[chosen])

            _add(chosen)
        else:
            _add(chosen)

        # If the chosen has an explicit map, try that next
        if chosen in model_map:
            _add(model_map[chosen])

        # If the original (pre-strip) appears in map, try that too
        rid = (requested_model_id or "").strip()
        if rid in model_map:
            _add(model_map[rid])

        # Then try configured fallback chain
        for m in chain:
            _add(m)

        # Optional: translate model ids to upstream api_identifier values
        if getattr(self.valves, "USE_API_IDENTIFIER", False):
            converted: List[str] = []
            for c in candidates:
                cc = self._to_api_identifier(c)
                if cc not in converted:
                    converted.append(cc)
            candidates = converted

        return candidates, extended_thinking

    @staticmethod
    def _looks_like_unsupported_model_error(status_code: int, text: str) -> bool:
        if status_code not in (400, 404):
            return False
        t = (text or "").lower()
        # Common proxy error strings
        return ("model" in t) and (
            "not support" in t or "not supported" in t or "unsupported" in t
        )

    def _post_with_model_fallback(
        self, url: str, headers: Dict[str, str], payload: Dict[str, Any], stream: bool
    ) -> requests.Response:
        """POST to Anthropic endpoint, retrying with fallback models on unsupported-model errors."""
        candidates = payload.get("_model_candidates")
        if not isinstance(candidates, list) or not candidates:
            # Default: single attempt
            candidates = [payload.get("model")]

        enable_fb = bool(getattr(self.valves, "ENABLE_MODEL_FALLBACK", True))

        last_resp: Optional[requests.Response] = None
        for idx, model_name in enumerate(candidates):
            p = dict(payload)
            p.pop("_model_candidates", None)
            if model_name:
                p["model"] = model_name

            if idx > 0:
                self._log("warning", f"Retrying with fallback model: {model_name}")

            resp = requests.post(
                url,
                headers=headers,
                json=p,
                stream=stream,
                timeout=(
                    getattr(self.valves, "CONNECT_TIMEOUT", 10),
                    getattr(self.valves, "READ_TIMEOUT", 300),
                ),
            )
            last_resp = resp

            if resp.status_code == 200:
                # Put chosen model back for logging/debugging downstream if needed
                payload["model"] = model_name
                return resp

            # Only retry on explicit unsupported-model errors
            if not enable_fb or not self._looks_like_unsupported_model_error(
                resp.status_code, resp.text
            ):
                return resp

            if getattr(self.valves, "DEBUG_LOGGING", False):
                try:
                    preview = (resp.text or "")[:300].replace("\n", " ")
                    self._log(
                        "debug",
                        f"Unsupported model response for model={model_name} status={resp.status_code} body_preview={preview}",
                    )
                except Exception:
                    pass

            # Otherwise continue to next candidate
            try:
                resp.close()
            except Exception:
                pass

        # No success; return the last response
        if last_resp is None:
            raise AnthropicAPIError(500, "No response")
        return last_resp

    def _select_model(self, requested_model_id: str) -> Tuple[str, bool]:
        """
        Select model name to send upstream.

        IMPORTANT: Different Anthropic-compatible proxies use different model naming.
        To maximize compatibility, we *do not* map to internal aliases here.
        We simply normalize common OpenWebUI prefixes/suffixes and pass through.
        """
        rid = (requested_model_id or "").strip()

        # Common OpenWebUI / provider prefixes
        if rid.startswith("anthropic."):
            rid = rid[len("anthropic.") :]
        if rid.startswith("anthropic/"):
            rid = rid[len("anthropic/") :]

        # If caller uses a path-like model id, keep only the last segment
        model_short_name = rid.split("/", 1)[-1] if "/" in rid else rid

        # Extended thinking suffix used by this Pipe
        extended_thinking = model_short_name.endswith("-extended-thinking")
        if extended_thinking:
            model_short_name = model_short_name[: -len("-extended-thinking")]

        # If still empty, fall back to original
        chosen = model_short_name or (requested_model_id or "claude-4-5-sonnet-latest")

        # Map to upstream API identifier (e.g., "claude-sonnet-4-5") when enabled.
        if getattr(self.valves, "USE_API_IDENTIFIER", True):
            chosen_api = self._to_api_identifier(chosen)
        else:
            chosen_api = chosen

        return chosen_api, extended_thinking

    def _prepare_payload(
        self,
        body: Dict[str, Any],
        processed_messages: List[Dict[str, Any]],
        chosen_model: str,
        extended_thinking: bool,
        system_message: Optional[str],
        anthropic_tools: Optional[List[Dict[str, Any]]],
        anthropic_tool_choice: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build the payload for the API request based on the processed messages and model selection.
        """
        payload: Dict[str, Any] = {
            "model": chosen_model,
            "messages": processed_messages,
            "stream": body.get("stream", False),
        }

        # stop sequences
        if "stop" in body:
            payload["stop_sequences"] = body["stop"]

        # Tool config (Anthropic Messages API)
        if anthropic_tools:
            payload["tools"] = anthropic_tools
            # Some Anthropic-compatible proxies behave better when tool_choice is explicit.
            payload["tool_choice"] = anthropic_tool_choice or {"type": "auto"}
        # Extended thinking models
        if extended_thinking:
            max_tokens = body.get("max_tokens", 20000)
            payload["max_tokens"] = max_tokens

            # IMPORTANT:
            # Anthropic "extended thinking" requires signed thinking blocks to be preserved across turns.
            # By default OpenWebUI's OpenAI-style chat format does not carry Anthropic thinking signatures.
            # We can still support thinking+tools by caching the signed thinking block keyed by tool_call_id
            # and replaying it on the subsequent tool_result request.
            allow_with_tools = bool(
                getattr(self.valves, "ALLOW_THINKING_WITH_TOOLS", False)
            )
            if anthropic_tools and (not allow_with_tools):
                logger.warning(
                    "extended thinking requested, but tools are enabled; set ANTHROPIC_ALLOW_THINKING_WITH_TOOLS=true to enable (requires signed-thinking replay)"
                )
            else:
                budget_tokens = min(16000, max_tokens - 1)
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                }

                # OpenWebUI docs warn that some forced tool_choice types are incompatible with extended thinking.
                # If user forced a tool call, fall back to auto for safety.
                if isinstance(payload.get("tool_choice"), dict) and payload[
                    "tool_choice"
                ].get("type") in ("any", "tool"):
                    logger.warning(
                        "extended thinking model: overriding tool_choice(any/tool) -> auto for compatibility"
                    )
                    payload["tool_choice"] = {"type": "auto"}

            for param in ["temperature", "top_k", "top_p"]:
                if param in body:
                    payload[param] = body[param]
        else:
            for param, default in [
                ("max_tokens", 4096),
                ("temperature", 0.8),
                ("top_k", 40),
                ("top_p", 0.9),
            ]:
                payload[param] = body.get(param, default)

        # Claude 4.5 models (and some proxies) reject specifying BOTH `temperature` and `top_p`.
        # OpenWebUI may send both (or we may set defaults), so normalize to a single knob.
        try:
            _m = str(payload.get("model", ""))
            _needs_single_sampling_param = ("-4-5" in _m) or ("claude-4-5" in _m)
            if (
                _needs_single_sampling_param
                and ("temperature" in payload)
                and ("top_p" in payload)
            ):
                _temp_user = ("temperature" in body) and (
                    body.get("temperature") is not None
                )
                _topp_user = ("top_p" in body) and (body.get("top_p") is not None)

                if _topp_user and not _temp_user:
                    # User intentionally set top_p -> drop temperature (default or user unset).
                    payload.pop("temperature", None)
                    logger.debug(
                        f"Normalized sampling params: removed temperature (keep top_p) for model={_m}"
                    )
                else:
                    # Default / temperature-only / both specified -> keep temperature, drop top_p.
                    payload.pop("top_p", None)
                    logger.debug(
                        f"Normalized sampling params: removed top_p (keep temperature) for model={_m}"
                    )
        except Exception:
            # Never fail the request due to normalization logic
            pass

        # If tools are enabled, add a short, explicit instruction to use exact tool names.
        if anthropic_tools:
            try:
                _names = [
                    t.get("name")
                    for t in anthropic_tools
                    if isinstance(t, dict) and t.get("name")
                ]
                if _names:
                    tool_hint = (
                        "You have access to external tools (function calling).\n"
                        "IMPORTANT: Tool names are identifiers. You MUST use an exact tool name from the list below (case-sensitive).\n"
                        'Do NOT translate tool names, do NOT paraphrase them, and do NOT invent new tool names (e.g., "专业中文网页搜索工具").\n'
                        "If you need a tool, call it via the native tool call mechanism, not by writing the tool name in plain text.\n"
                        f"Available tool names: {', '.join(_names)}.\n"
                        "——\n"
                        "你可以使用外部工具（函数调用）。\n"
                        "重要：工具名称是标识符，必须严格使用下面列表中的‘原样名称’（区分大小写）。\n"
                        '不要翻译、不要改写、不要杜撰工具名（例如："专业中文网页搜索工具"）。\n'
                        "需要工具时必须用原生工具调用发起，而不是在文本里说‘我要调用某某工具’。"
                    )
                    system_message = (
                        (str(system_message).strip() + "\n\n" + tool_hint)
                        if system_message
                        else tool_hint
                    )
            except Exception:
                # Never fail the request due to a hint string
                pass

        if system_message:
            payload["system"] = str(system_message)

        return payload

    # ---------------------------
    # OpenAI-compat response builders
    # ---------------------------

    @staticmethod
    def _openai_response_object(
        message: Dict[str, Any],
        model: str,
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
    ) -> Dict[str, Any]:
        created = int(time.time())
        usage = None
        if prompt_tokens is not None and completion_tokens is not None:
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

        res: Dict[str, Any] = {
            "id": f"chatcmpl_{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": message.get("_finish_reason", "stop"),
                }
            ],
        }
        if usage is not None:
            res["usage"] = usage
        # remove internal
        res["choices"][0]["message"].pop("_finish_reason", None)
        return res

    @staticmethod
    def _openai_sse_line(obj: Dict[str, Any]) -> str:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    # ---------------------------
    # Token usage (fallback)
    # ---------------------------

    def _get_tiktoken_encoder(self, model_hint: Optional[str] = None):
        """Return a tiktoken encoder (or None if tiktoken isn't available)."""
        if tiktoken is None:
            return None
        enc_name = (
            getattr(self.valves, "TIKTOKEN_ENCODING", None) or "cl100k_base"
        ).strip()
        # Try model-specific encoding first (only works for some OpenAI model names).
        if model_hint:
            try:
                return tiktoken.encoding_for_model(model_hint)  # type: ignore[attr-defined]
            except Exception:
                pass
        try:
            return tiktoken.get_encoding(enc_name)  # type: ignore[attr-defined]
        except Exception:
            # Last resort
            try:
                return tiktoken.get_encoding("cl100k_base")  # type: ignore[attr-defined]
            except Exception:
                return None

    @staticmethod
    def _sanitize_for_token_count(obj: Any) -> Any:
        """Remove/shorten large binary/base64 fields (e.g., image data) to avoid wildly inflated counts."""
        try:
            if isinstance(obj, dict):
                # Common pattern: {"type":"base64", "data":"..."}
                if (
                    "data" in obj
                    and isinstance(obj.get("data"), str)
                    and len(obj["data"]) > 200
                ):
                    # Only sanitize if it looks large enough to be base64-ish.
                    obj2 = dict(obj)
                    obj2["data"] = "<omitted>"
                    return {
                        k: Pipe._sanitize_for_token_count(v) for k, v in obj2.items()
                    }
                return {k: Pipe._sanitize_for_token_count(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [Pipe._sanitize_for_token_count(v) for v in obj]
            if isinstance(obj, str):
                # Truncate extremely long strings (tool schemas can be big; keep some head/tail)
                if len(obj) > 20000:
                    return obj[:10000] + "…<truncated>…" + obj[-2000:]
                return obj
            return obj
        except Exception:
            return obj

    def _estimate_tokens_from_obj(
        self, obj: Any, model_hint: Optional[str] = None
    ) -> Optional[int]:
        """Approximate token count by encoding a compact JSON representation."""
        if not getattr(self.valves, "FALLBACK_TIKTOKEN", True):
            return None
        enc = self._get_tiktoken_encoder(model_hint)
        if enc is None:
            return None
        try:
            safe_obj = self._sanitize_for_token_count(obj)
            s = json.dumps(safe_obj, ensure_ascii=False, separators=(",", ":"))
            return len(enc.encode(s))
        except Exception as e:
            logger.debug(f"Token estimate failed: {e}")
            return None

    def _estimate_tokens_from_text(
        self, text: str, model_hint: Optional[str] = None
    ) -> Optional[int]:
        if not getattr(self.valves, "FALLBACK_TIKTOKEN", True):
            return None
        enc = self._get_tiktoken_encoder(model_hint)
        if enc is None:
            return None
        try:
            return len(enc.encode(text or ""))
        except Exception as e:
            logger.debug(f"Token estimate failed: {e}")
            return None

    # ---------------------------
    # Pipe entry point
    # ---------------------------

    def pipe(
        self, body: Dict[str, Any]
    ) -> Union[str, Dict[str, Any], Generator[str, None, None]]:
        if not self.valves.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required but not provided")
        if "model" not in body:
            raise ValueError("Model must be specified in the request body")
        if "messages" not in body:
            raise ValueError("Messages must be specified in the request body")

        # Apply latest valves (UI overrides) on every request
        self._configure_logging()
        new_endpoint = self._resolve_messages_endpoint(self.valves.BASE_URL)
        if getattr(self, "api_endpoint", None) != new_endpoint:
            self.api_endpoint = new_endpoint
        try:
            self._log("info", f"Using Anthropic Messages endpoint: {self.api_endpoint}")
            ct = getattr(self.valves, "CONNECT_TIMEOUT", 10)
            rt = getattr(self.valves, "READ_TIMEOUT", 300)
            self._log("debug", f"HTTP timeouts connect={ct}s read={rt}s")
        except Exception:
            pass

        # Debug: tool inventory coming from OpenWebUI Native Tools
        try:
            _tools = body.get("tools") or []
            if isinstance(_tools, list) and _tools:
                tool_names = []
                for t in _tools:
                    if isinstance(t, dict):
                        fn = t.get("function") or {}
                        tool_names.append(fn.get("name") or t.get("name") or "<?>")
                self._log("debug", f"Incoming OpenAI tools: {tool_names}")
            else:
                self._log("debug", "Incoming OpenAI tools: []")
        except Exception as _e:
            self._log("debug", f"Failed to introspect tools: {_e}")

        system_message, messages = pop_system_message(body["messages"])
        is_tool_continuation = any(
            isinstance(m, dict) and m.get("role") == "tool" for m in (messages or [])
        )
        model_candidates, extended_thinking = self._candidate_models_for_request(
            body["model"]
        )
        chosen_model = (
            model_candidates[0] if model_candidates else "claude-3-7-sonnet-latest"
        )
        requested_openai_model = body.get("model", chosen_model)
        actual_openai_model = requested_openai_model
        try:
            if isinstance(requested_openai_model, str):
                if "." in requested_openai_model:
                    prefix = requested_openai_model.split(".", 1)[0]
                    actual_openai_model = f"{prefix}.{chosen_model}"
                else:
                    actual_openai_model = chosen_model
        except Exception:
            actual_openai_model = requested_openai_model

        # NOTE: do not reference `payload` here (it is built later). Use `model_candidates` instead.
        show_candidates = (
            list(model_candidates) if isinstance(model_candidates, list) else []
        )
        if getattr(self.valves, "USE_API_IDENTIFIER", False) and isinstance(
            show_candidates, list
        ):
            show_candidates = [self._to_api_identifier(c) for c in show_candidates]
        upstream_first = show_candidates[0] if show_candidates else actual_openai_model
        self._log(
            "info",
            f"Request model={body.get('model')} chosen={actual_openai_model} upstream={upstream_first} stream={bool(body.get('stream', False))}",
        )
        if getattr(self.valves, "DEBUG_LOGGING", False):
            try:
                self._log("debug", f"Model candidates: {show_candidates}")
            except Exception:
                pass

        openai_tools = body.get("tools") or body.get("functions")  # be tolerant
        anthropic_tools = self._openai_tools_to_anthropic(openai_tools)
        anthropic_tool_choice = self._openai_tool_choice_to_anthropic(
            body.get("tool_choice")
        )

        valid_tool_names = None
        if anthropic_tools:
            valid_tool_names = {t.get("name") for t in anthropic_tools if t.get("name")}
            self._log("debug", f"Tools enabled: {len(anthropic_tools)}")

        processed_messages, _, saw_tools_in_messages = (
            self._convert_messages_for_anthropic(messages, valid_tool_names)
        )
        native_tool_mode = bool(anthropic_tools) or saw_tools_in_messages

        headers = {
            "x-api-key": self.valves.ANTHROPIC_API_KEY,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }

        payload = self._prepare_payload(
            body,
            processed_messages,
            chosen_model,
            extended_thinking,
            system_message,
            anthropic_tools,
            anthropic_tool_choice,
        )
        # Provide candidate models to the request helper so we can retry on unsupported-model errors.
        # This is removed before the actual request is sent.
        payload["_model_candidates"] = model_candidates
        self._log("debug", f"Payload keys: {list(payload.keys())}")

        try:
            if body.get("stream", False):
                if native_tool_mode:
                    return self.stream_response_openai(
                        self.api_endpoint,
                        headers,
                        payload,
                        model_for_openai=actual_openai_model,
                        valid_tool_names=valid_tool_names,
                        is_tool_continuation=is_tool_continuation,
                    )
                return self.stream_response_text(self.api_endpoint, headers, payload)
            else:
                if native_tool_mode:
                    return self.non_stream_response_openai(
                        self.api_endpoint,
                        headers,
                        payload,
                        model_for_openai=actual_openai_model,
                        valid_tool_names=valid_tool_names,
                        is_tool_continuation=is_tool_continuation,
                    )
                return self.non_stream_response_text(
                    self.api_endpoint, headers, payload
                )
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise AnthropicAPIError(500, f"Request failed: {e}")
        except Exception as e:
            logger.error(f"Error in pipe method: {e}")
            raise AnthropicAPIError(500, f"Error: {e}")

    def _extract_usage_tokens(self, obj: Any) -> Tuple[Optional[int], Optional[int]]:
        """Best-effort extract (prompt_tokens, completion_tokens) from Anthropic or proxy responses.

        Supports:
        - Anthropic Messages: {"usage":{"input_tokens":..,"output_tokens":..}}
        - Some proxies/OpenAI-like: {"usage":{"prompt_tokens":..,"completion_tokens":..}}
        - Top-level keys: input_tokens/output_tokens or prompt_tokens/completion_tokens
        - Wrapped shapes: {"message":{...,"usage":{...}}}
        """
        if not isinstance(obj, dict):
            return (None, None)

        # Some providers wrap a message object
        base = obj
        if isinstance(obj.get("message"), dict):
            # Keep both; prefer explicit top-level usage if present
            base = obj.get("message") or obj

        def _get_int(d: Any, key: str) -> Optional[int]:
            if isinstance(d, dict):
                v = d.get(key)
                return v if isinstance(v, int) else None
            return None

        usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else None
        if usage is None and isinstance(base, dict):
            usage = base.get("usage") if isinstance(base.get("usage"), dict) else None

        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None

        # Preferred: usage dict
        if isinstance(usage, dict):
            prompt_tokens = _get_int(usage, "input_tokens") or _get_int(
                usage, "prompt_tokens"
            )
            completion_tokens = _get_int(usage, "output_tokens") or _get_int(
                usage, "completion_tokens"
            )

        # Fallback: top-level keys
        if prompt_tokens is None:
            prompt_tokens = _get_int(obj, "input_tokens") or _get_int(
                obj, "prompt_tokens"
            )
        if completion_tokens is None:
            completion_tokens = _get_int(obj, "output_tokens") or _get_int(
                obj, "completion_tokens"
            )

        # Some proxies only include totals; we intentionally do NOT infer prompt/completion split from total.
        return (prompt_tokens, completion_tokens)

    # ---------------------------
    # Streaming (text-only) legacy
    # ---------------------------

    def stream_response_text(
        self, url: str, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """
        Existing behavior: yield plain text chunks (plus <think> tags for extended thinking).
        """
        thinking_state = ThinkingState.NOT_STARTED
        saw_block_stream: bool = False

        try:
            with self._post_with_model_fallback(
                url, headers, payload, stream=True
            ) as response:
                if response.status_code != 200:
                    raise AnthropicAPIError(response.status_code, response.text)

                for line in response.iter_lines():
                    if not line:
                        continue
                    decoded_line = line.decode("utf-8")
                    if not decoded_line.startswith("data: "):
                        continue

                    raw_json = decoded_line[6:]
                    if raw_json.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(raw_json)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON: {raw_json}")
                        continue

                    event_type = data.get("type")
                    if event_type == EventType.CONTENT_BLOCK_START.value:
                        saw_block_stream = True
                        content_block = data.get("content_block", {})
                        if content_block.get("type") == ContentType.TEXT.value:
                            if thinking_state == ThinkingState.IN_PROGRESS:
                                yield "</think>\n"
                                thinking_state = ThinkingState.NOT_STARTED
                            yield content_block.get("text", "")
                        continue

                    elif event_type == EventType.CONTENT_BLOCK_DELTA.value:
                        saw_block_stream = True
                        delta = data.get("delta", {})
                        delta_type = delta.get("type")
                        if delta_type == DeltaType.THINKING_DELTA.value:
                            if thinking_state == ThinkingState.NOT_STARTED:
                                yield "<think>"
                                thinking_state = ThinkingState.IN_PROGRESS
                            yield delta.get("thinking", "")
                        elif delta_type == DeltaType.TEXT_DELTA.value:
                            if thinking_state == ThinkingState.IN_PROGRESS:
                                yield "</think>\n"
                                thinking_state = ThinkingState.NOT_STARTED
                            yield delta.get("text", "")

                    elif event_type == EventType.MESSAGE.value:
                        if saw_block_stream:
                            # Avoid duplicating content for providers that send both block-level stream events and a full message event.
                            continue
                        for content in data.get("content", []):
                            content_type = content.get("type")
                            if content_type == ContentType.THINKING.value:
                                if thinking_state == ThinkingState.NOT_STARTED:
                                    yield "<think>"
                                    thinking_state = ThinkingState.IN_PROGRESS
                                yield content.get("text", "")
                            elif content_type == ContentType.TEXT.value:
                                if thinking_state == ThinkingState.IN_PROGRESS:
                                    yield "</think>\n"
                                    thinking_state = ThinkingState.NOT_STARTED
                                yield content.get("text", "")

                    elif event_type == EventType.MESSAGE_STOP.value:
                        if thinking_state == ThinkingState.IN_PROGRESS:
                            yield "</think>"
                        break

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise AnthropicAPIError(500, f"Request failed: {e}")
        except Exception as e:
            logger.error(f"General error in stream_response_text method: {e}")
            raise AnthropicAPIError(500, f"Error: {e}")

    # ---------------------------
    # Streaming (OpenAI SSE) with tool calling
    # ---------------------------

    def stream_response_openai(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model_for_openai: str,
        valid_tool_names: Optional[set] = None,
        is_tool_continuation: bool = False,
    ) -> Generator[str, None, None]:
        """
        Convert Anthropic SSE events into OpenAI chat.completion.chunk SSE.
        Supports tool_use -> tool_calls (function calling) conversion.
        """
        thinking_state = ThinkingState.NOT_STARTED
        saw_tool_use = False

        # Accumulators for signed thinking replay (extended thinking + tool calling)
        thinking_text_accum: List[str] = []
        thinking_signature_accum: List[str] = []
        tool_ids_seen: List[str] = []

        tool_stream_mode = (
            (getattr(self.valves, "TOOL_CALL_STREAM_MODE", "single") or "single")
            .strip()
            .lower()
        )
        if tool_stream_mode not in ("single", "delta"):
            tool_stream_mode = "single"

        emitted_tool_blocks: set[int] = set()

        # Usage tracking
        api_prompt_tokens: Optional[int] = None
        api_completion_tokens: Optional[int] = None
        completion_text_accum: List[str] = []
        tool_args_accum_by_block: Dict[int, str] = {}
        pending_text_parts: List[str] = []
        saw_block_stream: bool = False

        openai_id = f"chatcmpl_{uuid.uuid4().hex}"
        created = int(time.time())
        # Buffer for converted SSE lines
        nonlocal_gen: List[str] = []
        # Map Anthropic content_block index -> OpenAI tool_calls index
        tool_index_by_block: Dict[int, int] = {}
        tool_id_by_block: Dict[int, str] = {}
        tool_name_by_block: Dict[int, str] = {}

        # Emit the initial role chunk (helps some clients)
        first_chunk = {
            "id": openai_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_for_openai,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
        yield self._openai_sse_line(first_chunk)
        def emit_content_delta(text: str) -> None:
            """Emit normal assistant text.

            Hard-coded behavior for OpenWebUI tool loops (stable UI):
            - We always buffer normal assistant text until we know whether this assistant message
              will call tools.
            - If a tool_use happens in this assistant message: buffered text is emitted as
              `delta.reasoning_content` (thinking panel), not `delta.content`.
            - If no tool_use happens: buffered text is emitted as normal `delta.content` at MESSAGE_STOP.

            This makes tool-calling turns clean (only tool_calls + thinking) and prevents "analysis"
            lines from showing up as normal content before the final answer.
            """
            if not text:
                return

            # If this turn is confirmed to be a tool-calling turn, keep all text inside reasoning.
            if saw_tool_use:
                emit_thinking_delta(text)
                return

            # Otherwise, buffer until we know whether tools will be used in this assistant message.
            pending_text_parts.append(text)

        def emit_thinking_delta(text: str) -> None:
            """Emit Claude 'thinking' as OpenAI-style reasoning deltas.

            Why: OpenWebUI has better native support for interleaved thinking during tool loops
            when reasoning is streamed via `delta.reasoning_content` (instead of embedding
            `<think>...</think>` inside `delta.content`). This also avoids various tag-parsing
            glitches that can cause hidden blocks to suddenly burst into the main text.
            """
            if text == "":
                return
            completion_text_accum.append(text or "")
            chunk = {
                "id": openai_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_for_openai,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "", "reasoning_content": text},
                        "finish_reason": None,
                    }
                ],
            }
            nonlocal_gen.append(self._openai_sse_line(chunk))

        def _thinking_open_marker() -> str:
            """Deprecated: we don't emit <think> tags in streaming mode.

            OpenWebUI can consume `delta.reasoning_content` directly, which is more stable
            during native tool loops.
            """
            return ""

        def _thinking_close_marker() -> str:
            """Deprecated: we don't emit </think> tags in streaming mode."""
            return ""

        def _format_thinking_text(t: str) -> str:
            """Format thinking delta text for visibility."""
            if not t:
                return ""
            # Keep raw thinking text; avoid extra prefixes in streaming to reduce flicker.
            return t

        def emit_tool_start(block_index: int, tool_id: str, tool_name: str) -> None:
            nonlocal saw_tool_use
            # Deduplicate: some providers may emit repeated content_block_start for the same tool.
            if block_index in tool_index_by_block:
                return

            tool_args_accum_by_block.setdefault(block_index, "")

            # If we buffered any normal text before we knew a tool would be used,
            # push it into the reasoning panel instead of dropping it.
            if pending_text_parts:
                emit_thinking_delta("".join(pending_text_parts))
                pending_text_parts.clear()

            saw_tool_use = True

            tc_index = len(tool_index_by_block)
            tool_index_by_block[block_index] = tc_index
            tool_id_by_block[block_index] = tool_id
            tool_name_by_block[block_index] = tool_name
            if isinstance(tool_id, str) and tool_id:
                tool_ids_seen.append(tool_id)

            # In "single" mode, delay emitting tool_calls until we have full arguments.
            if tool_stream_mode == "single":
                return

            chunk = {
                "id": openai_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_for_openai,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "index": tc_index,
                                    "id": tool_id,
                                    "type": "function",
                                    "function": {"name": tool_name, "arguments": ""},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            }
            nonlocal_gen.append(self._openai_sse_line(chunk))
            self._log(
                "info",
                f"TOOL_CALL→OWUI start idx={tc_index} name={tool_name} id={tool_id}",
            )

        def emit_tool_args(block_index: int, partial_json: str) -> None:
            # Accumulate full JSON arguments (needed for one-shot emission and token accounting)
            tool_args_accum_by_block[block_index] = tool_args_accum_by_block.get(
                block_index, ""
            ) + (partial_json or "")

            if tool_stream_mode == "single":
                return

            if block_index not in tool_index_by_block:
                return

            tc_index = tool_index_by_block[block_index]
            tool_id = tool_id_by_block.get(block_index, "")

            chunk = {
                "id": openai_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_for_openai,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "index": tc_index,
                                    "id": tool_id,
                                    "type": "function",
                                    "function": {"arguments": partial_json or ""},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            }
            nonlocal_gen.append(self._openai_sse_line(chunk))

        try:
            with self._post_with_model_fallback(
                url, headers, payload, stream=True
            ) as response:
                if response.status_code != 200:
                    raise AnthropicAPIError(response.status_code, response.text)

                for line in response.iter_lines():
                    # flush buffered converted lines
                    while nonlocal_gen:
                        yield nonlocal_gen.pop(0)

                    if not line:
                        continue
                    decoded_line = line.decode("utf-8")
                    if not decoded_line.startswith("data: "):
                        continue

                    raw_json = decoded_line[6:]
                    if raw_json.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(raw_json)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON: {raw_json}")
                        continue

                    event_type = data.get("type")
                    if event_type in (EventType.PING.value,):
                        continue

                    # Capture token usage (if provided by upstream).
                    # Per Anthropic streaming docs, usage in message_delta is cumulative.
                    if event_type == EventType.MESSAGE_START.value:
                        msg0 = data.get("message") or {}
                        pt, ct = self._extract_usage_tokens(msg0)
                        if isinstance(pt, int) and api_prompt_tokens is None:
                            api_prompt_tokens = pt
                        if isinstance(ct, int):
                            api_completion_tokens = ct
                        continue

                    if event_type == EventType.MESSAGE_DELTA.value:
                        pt, ct = self._extract_usage_tokens(data)
                        if isinstance(pt, int) and api_prompt_tokens is None:
                            api_prompt_tokens = pt
                        if isinstance(ct, int):
                            api_completion_tokens = ct
                        continue

                    # NOTE: Do NOT `continue` on MESSAGE_STOP here.
                    # We still want the downstream MESSAGE_STOP handler (later in this loop)
                    # to emit finish_reason + [DONE] and to flush any buffered content.
                    if event_type == EventType.MESSAGE_STOP.value:
                        # Some providers attach final usage here
                        pt, ct = self._extract_usage_tokens(data)
                        if isinstance(pt, int) and api_prompt_tokens is None:
                            api_prompt_tokens = pt
                        if isinstance(ct, int):
                            api_completion_tokens = ct
                        # fallthrough

                    if event_type == EventType.ERROR.value:
                        err = data.get("error") or {}
                        raise AnthropicAPIError(
                            500, json.dumps(err, ensure_ascii=False)
                        )

                    if event_type == EventType.CONTENT_BLOCK_START.value:
                        saw_block_stream = True
                        idx = data.get("index")
                        content_block = data.get("content_block", {}) or {}

                        # Tool use block
                        if content_block.get(
                            "type"
                        ) == ContentType.TOOL_USE.value and isinstance(idx, int):
                            if thinking_state == ThinkingState.IN_PROGRESS:
                                emit_thinking_delta(_thinking_close_marker())
                                thinking_state = ThinkingState.NOT_STARTED
                            try:
                                self._log(
                                    "debug",
                                    f"saw tool_use content_block_start idx={idx} name={content_block.get('name')} id={content_block.get('id')}",
                                )
                            except Exception:
                                pass
                            _raw_tool_name = content_block.get("name")
                            _mapped_tool_name = (
                                self._best_effort_tool_name(
                                    _raw_tool_name, valid_tool_names
                                )
                                or ""
                            )[:200]
                            if (
                                getattr(self.valves, "LOG_ENABLED", False)
                                and _raw_tool_name
                                and _mapped_tool_name
                                and _raw_tool_name != _mapped_tool_name
                            ):
                                self._log(
                                    "warning",
                                    f"Tool name mismatch from model: raw={_raw_tool_name!r} mapped={_mapped_tool_name!r}",
                                )
                            emit_tool_start(
                                idx,
                                content_block.get("id") or f"toolu_{uuid.uuid4().hex}",
                                _mapped_tool_name,
                            )
                            # Some proxies may include full input object here; emit as one shot
                            if (
                                isinstance(content_block.get("input"), dict)
                                and content_block["input"]
                            ):
                                emit_tool_args(
                                    idx,
                                    json.dumps(
                                        content_block["input"], ensure_ascii=False
                                    ),
                                )
                            continue

                        # Text blocks (for completeness; deltas carry the actual text)
                        if content_block.get("type") == ContentType.TEXT.value:
                            # If we previously started thinking, close it before normal text output.
                            if thinking_state == ThinkingState.IN_PROGRESS:
                                emit_thinking_delta(_thinking_close_marker())
                                thinking_state = ThinkingState.NOT_STARTED
                            initial_text = content_block.get("text", "")
                            if initial_text:
                                emit_content_delta(initial_text)
                        continue

                    if event_type == EventType.CONTENT_BLOCK_STOP.value:
                        saw_block_stream = True
                        idx = data.get("index")
                        if tool_stream_mode == "single" and isinstance(idx, int):
                            tool_id = tool_id_by_block.get(idx)
                            tool_name = tool_name_by_block.get(idx)
                            tc_index = tool_index_by_block.get(idx)
                            if (
                                tool_id
                                and tool_name
                                and isinstance(tc_index, int)
                                and idx not in emitted_tool_blocks
                            ):
                                full_args = (
                                    tool_args_accum_by_block.get(idx, "") or "{}"
                                )
                                nonlocal_gen.append(
                                    self._openai_sse_line(
                                        {
                                            "id": openai_id,
                                            "object": "chat.completion.chunk",
                                            "created": created,
                                            "model": model_for_openai,
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {
                                                        "content": "",
                                                        "tool_calls": [
                                                            {
                                                                "index": tc_index,
                                                                "id": tool_id,
                                                                "type": "function",
                                                                "function": {
                                                                    "name": tool_name,
                                                                    "arguments": full_args,
                                                                },
                                                            }
                                                        ],
                                                    },
                                                    "finish_reason": None,
                                                }
                                            ],
                                        }
                                    )
                                )
                                emitted_tool_blocks.add(idx)
                                saw_tool_use = True
                                self._log(
                                    "info",
                                    f"TOOL_CALL→OWUI one-shot idx={tc_index} name={tool_name} id={tool_id} args_len={len(full_args)}",
                                )
                        continue

                    if event_type == EventType.CONTENT_BLOCK_DELTA.value:
                        saw_block_stream = True
                        idx = data.get("index")
                        delta = data.get("delta", {}) or {}
                        delta_type = delta.get("type")

                        if delta_type == DeltaType.THINKING_DELTA.value:
                            if thinking_state == ThinkingState.NOT_STARTED:
                                emit_thinking_delta(_thinking_open_marker())
                                thinking_state = ThinkingState.IN_PROGRESS
                            t = delta.get("thinking", "") or ""
                            thinking_text_accum.append(t)
                            emit_thinking_delta(_format_thinking_text(t))
                            continue

                        if delta_type == DeltaType.TEXT_DELTA.value:
                            if thinking_state == ThinkingState.IN_PROGRESS:
                                emit_thinking_delta(_thinking_close_marker())
                                thinking_state = ThinkingState.NOT_STARTED
                            emit_content_delta(delta.get("text", ""))
                            continue

                        if (
                            delta_type == DeltaType.INPUT_JSON_DELTA.value
                            and isinstance(idx, int)
                        ):
                            # partial_json is a partial JSON string per Anthropic streaming docs
                            emit_tool_args(idx, delta.get("partial_json", ""))
                            continue

                        if delta_type == DeltaType.SIGNATURE_DELTA.value:
                            # Signed thinking replay: capture signature so we can replay thinking+signature on tool_result.
                            thinking_signature_accum.append(
                                delta.get("signature", "") or ""
                            )
                            continue

                        # ignore other delta types

                    if event_type == EventType.MESSAGE.value:
                        # Some proxies send the full message as a single event (no block-level deltas).
                        # If we already saw block-level streaming events, skip to avoid duplication.
                        if saw_block_stream:
                            continue

                        for content in data.get("content", []):
                            if not isinstance(content, dict):
                                continue

                            ctype = content.get("type")

                            if ctype == ContentType.THINKING.value:
                                t = (
                                    content.get("thinking")
                                    or content.get("text")
                                    or ""
                                )
                                if t:
                                    thinking_text_accum.append(t)
                                    emit_thinking_delta(t)
                                sig = content.get("signature") or ""
                                if isinstance(sig, str) and sig:
                                    thinking_signature_accum.append(sig)
                                continue

                            if ctype == ContentType.TEXT.value:
                                emit_content_delta(content.get("text", "") or "")
                                continue

                            if ctype == ContentType.TOOL_USE.value:
                                # If we buffered normal text before the tool call,
                                # keep it in the reasoning panel (don't leak into main content).
                                if pending_text_parts:
                                    emit_thinking_delta("".join(pending_text_parts))
                                    pending_text_parts.clear()

                                saw_tool_use = True
                                tid = content.get("id")
                                if isinstance(tid, str) and tid:
                                    tool_ids_seen.append(tid)

                                # emit as a "complete" tool call
                                tool_id = (
                                    content.get("id") or f"toolu_{uuid.uuid4().hex}"
                                )
                                tool_name = content.get("name") or ""
                                args = content.get("input") or {}
                                # allocate next index
                                tc_index = len(tool_index_by_block)
                                chunk = {
                                    "id": openai_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_for_openai,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "content": "",
                                                "tool_calls": [
                                                    {
                                                        "index": tc_index,
                                                        "id": tool_id,
                                                        "type": "function",
                                                        "function": {
                                                            "name": tool_name,
                                                            "arguments": json.dumps(
                                                                args, ensure_ascii=False
                                                            ),
                                                        },
                                                    }
                                                ],
                                            },
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                nonlocal_gen.append(self._openai_sse_line(chunk))
                                continue

                        continue

                    if event_type == EventType.MESSAGE_STOP.value:
                        if thinking_state == ThinkingState.IN_PROGRESS:
                            emit_thinking_delta(_thinking_close_marker())
                            thinking_state = ThinkingState.NOT_STARTED

                        # In "single" mode, some proxies omit CONTENT_BLOCK_STOP.
                        # Emit any pending tool calls now.
                        if tool_stream_mode == "single" and tool_id_by_block:
                            for idx, tool_id in list(tool_id_by_block.items()):
                                if idx in emitted_tool_blocks:
                                    continue
                                tool_name = tool_name_by_block.get(idx)
                                tc_index = tool_index_by_block.get(idx)
                                if not tool_name or not isinstance(tc_index, int):
                                    continue
                                full_args = (
                                    tool_args_accum_by_block.get(idx, "") or "{}"
                                )
                                nonlocal_gen.append(
                                    self._openai_sse_line(
                                        {
                                            "id": openai_id,
                                            "object": "chat.completion.chunk",
                                            "created": created,
                                            "model": model_for_openai,
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {
                                                        "content": "",
                                                        "tool_calls": [
                                                            {
                                                                "index": tc_index,
                                                                "id": tool_id,
                                                                "type": "function",
                                                                "function": {
                                                                    "name": tool_name,
                                                                    "arguments": full_args,
                                                                },
                                                            }
                                                        ],
                                                    },
                                                    "finish_reason": None,
                                                }
                                            ],
                                        }
                                    )
                                )
                                emitted_tool_blocks.add(idx)
                                saw_tool_use = True
                                self._log(
                                    "info",
                                    f"TOOL_CALL→OWUI one-shot (msg_stop) idx={tc_index} name={tool_name} id={tool_id} args_len={len(full_args)}",
                                )

                        # If we used tools and the model produced signed thinking, cache it keyed by tool_call_id so we
                        # can replay the exact thinking+signature on the next request that contains tool_result.
                        if saw_tool_use and bool(
                            getattr(self.valves, "ALLOW_THINKING_WITH_TOOLS", False)
                        ):
                            thinking_text = "".join(thinking_text_accum).strip()
                            thinking_sig = "".join(thinking_signature_accum).strip()
                            # Prefer the tool ids we emitted to OpenWebUI (tool_id_by_block), fallback to collected ids.
                            tool_ids = [
                                tid
                                for tid in (
                                    list(tool_id_by_block.values()) or tool_ids_seen
                                )
                                if isinstance(tid, str) and tid
                            ]
                            # De-dup while preserving order
                            seen = set()
                            tool_ids = [
                                x for x in tool_ids if not (x in seen or seen.add(x))
                            ]
                            if thinking_text and thinking_sig and tool_ids:
                                self._cache_signed_thinking(
                                    tool_ids, thinking_text, thinking_sig
                                )
                            elif (
                                thinking_text
                                and tool_ids
                                and (not thinking_sig)
                                and getattr(self.valves, "LOG_ENABLED", False)
                            ):
                                logger.warning(
                                    "extended thinking+tools: missing thinking signature from stream; cannot replay signed thinking reliably"
                                )


                        # Flush any buffered normal text now.
                        # - Tool-calling turn: keep it in reasoning_content.
                        # - Normal turn: show it as normal content.
                        if pending_text_parts:
                            buffered = "".join(pending_text_parts)
                            pending_text_parts.clear()
                            if saw_tool_use:
                                emit_thinking_delta(buffered)
                            else:
                                completion_text_accum.append(buffered)
                                nonlocal_gen.append(
                                    self._openai_sse_line(
                                        {
                                            "id": openai_id,
                                            "object": "chat.completion.chunk",
                                            "created": created,
                                            "model": model_for_openai,
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {"content": buffered},
                                                    "finish_reason": None,
                                                }
                                            ],
                                        }
                                    )
                                )


                        finish_reason = "tool_calls" if saw_tool_use else "stop"

                        # Emit final finish_reason chunk.
                        #
                        # Open WebUI (some builds) has an edge-case: when finish_reason == 'tool_calls',
                        # it expects a tool_calls delta in the final chunk; otherwise it can raise:
                        #   cannot access local variable 'tool_content' where it is not associated with a value
                        #
                        # To avoid duplicating tool name/arguments (some assemblers concatenate string fields),
                        # we emit placeholder tool_calls entries with empty name/arguments but correct indices/ids.
                        final_delta = {"content": ""}
                        if finish_reason == "tool_calls":
                            tool_calls_placeholder = []
                            for bidx in sorted(
                                emitted_tool_blocks or tool_id_by_block.keys()
                            ):
                                tci = tool_index_by_block.get(bidx)
                                tid = tool_id_by_block.get(bidx)
                                if (
                                    isinstance(tci, int)
                                    and isinstance(tid, str)
                                    and tid
                                ):
                                    tool_calls_placeholder.append(
                                        {
                                            "index": tci,
                                            "id": tid,
                                            "type": "function",
                                            "function": {"arguments": ""},
                                        }
                                    )
                            if tool_calls_placeholder:
                                final_delta["tool_calls"] = tool_calls_placeholder
                        final_chunk = {
                            "id": openai_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_for_openai,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": final_delta,
                                    "finish_reason": finish_reason,
                                }
                            ],
                        }
                        nonlocal_gen.append(self._openai_sse_line(final_chunk))

                        # Emit usage chunk (OpenAI-style) if enabled
                        if getattr(self.valves, "RETURN_TOKENS", True):
                            prompt_tokens = api_prompt_tokens
                            completion_tokens = api_completion_tokens

                            if prompt_tokens is None:
                                prompt_tokens = self._estimate_tokens_from_obj(
                                    payload, model_hint=model_for_openai
                                )
                            if completion_tokens is None:
                                # Count the output we streamed to OpenWebUI (text + tool args)
                                out_text = "".join(completion_text_accum)
                                out_tools = (
                                    json.dumps(
                                        tool_args_accum_by_block,
                                        ensure_ascii=False,
                                        separators=(",", ":"),
                                    )
                                    if tool_args_accum_by_block
                                    else ""
                                )
                                completion_tokens = self._estimate_tokens_from_text(
                                    out_text + out_tools, model_hint=model_for_openai
                                )

                            if getattr(self.valves, "LOG_ENABLED", False):
                                logger.info(
                                    f"Token usage (stream): prompt={prompt_tokens} completion={completion_tokens}"
                                )

                            if (
                                prompt_tokens is not None
                                and completion_tokens is not None
                            ):
                                usage_chunk = {
                                    "id": openai_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_for_openai,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": ""},
                                            "finish_reason": None,
                                        }
                                    ],
                                    "usage": {
                                        "prompt_tokens": int(prompt_tokens),
                                        "completion_tokens": int(completion_tokens),
                                        "total_tokens": int(prompt_tokens)
                                        + int(completion_tokens),
                                    },
                                }
                                nonlocal_gen.append(self._openai_sse_line(usage_chunk))

                        nonlocal_gen.append("data: [DONE]\n\n")
                        break

                # flush remaining
                while nonlocal_gen:
                    yield nonlocal_gen.pop(0)

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise AnthropicAPIError(500, f"Request failed: {e}")
        except Exception as e:
            logger.error(f"General error in stream_response_openai method: {e}")
            raise

    # ---------------------------
    # Non-streaming (text-only) legacy
    # ---------------------------

    def non_stream_response_text(
        self, url: str, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> str:
        """
        Handle non-streaming responses from the API and return the final text output.
        """
        try:
            response = self._post_with_model_fallback(
                url, headers, payload, stream=False
            )
            if response.status_code != 200:
                raise AnthropicAPIError(response.status_code, response.text)

            res = response.json()

            # Process response content
            if isinstance(res.get("content"), list):
                text_parts, thinking_parts = [], []
                for item in res["content"]:
                    if item.get("type") == ContentType.TEXT.value:
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == ContentType.THINKING.value:
                        thinking_parts.append(item.get("text", ""))
                if thinking_parts:
                    return f"<think>{''.join(thinking_parts)}</think>\n{''.join(text_parts)}"
                elif text_parts:
                    return "".join(text_parts)

            message = res.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), list):
                text_parts, thinking_parts = [], []
                for item in message["content"]:
                    if item.get("type") == ContentType.TEXT.value:
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == ContentType.THINKING.value:
                        thinking_parts.append(item.get("text", ""))
                if thinking_parts:
                    return f"<think>{''.join(thinking_parts)}</think>\n{''.join(text_parts)}"
                elif text_parts:
                    return "".join(text_parts)
            elif isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]

            if "text" in res:
                return res["text"]

            logger.warning(f"Could not extract text from response: {res}")
            return str(res)

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed non-stream request: {e}")
            raise AnthropicAPIError(500, f"Request failed: {e}")
        except Exception as e:
            logger.error(f"General error in non_stream_response_text method: {e}")
            raise

    # ---------------------------
    # Non-streaming (OpenAI response) with tool calling
    # ---------------------------

    def non_stream_response_openai(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model_for_openai: str,
    ) -> Dict[str, Any]:
        """
        Convert Anthropic non-stream response into OpenAI chat.completion response.
        Supports tool_use -> tool_calls conversion.
        """
        response = self._post_with_model_fallback(url, headers, payload, stream=False)
        if response.status_code != 200:
            raise AnthropicAPIError(response.status_code, response.text)

        res = response.json()

        # Some proxies wrap in {"message":{...}}
        msg = res.get("message") if isinstance(res.get("message"), dict) else res

        content_blocks = msg.get("content") if isinstance(msg, dict) else None
        stop_reason = msg.get("stop_reason") if isinstance(msg, dict) else None

        text_parts: List[str] = []
        thinking_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        if isinstance(content_blocks, list):
            for b in content_blocks:
                if not isinstance(b, dict):
                    continue
                btype = b.get("type")
                if btype == ContentType.TEXT.value:
                    text_parts.append(b.get("text", ""))
                elif btype == ContentType.THINKING.value:
                    thinking_parts.append(b.get("thinking") or b.get("text") or "")
                elif btype == ContentType.TOOL_USE.value:
                    tool_calls.append(
                        {
                            "id": b.get("id") or f"toolu_{uuid.uuid4().hex}",
                            "type": "function",
                            "function": {
                                "name": b.get("name") or "",
                                "arguments": json.dumps(
                                    b.get("input") or {}, ensure_ascii=False
                                ),
                            },
                        }
                    )

        # If this turn contains tool calls, suppress normal assistant text in this message.
        # The follow-up turn (after tools) will contain the user-visible answer.
        content_text = "" if tool_calls else "".join(text_parts)
        finish_reason = (
            "tool_calls" if tool_calls or stop_reason == "tool_use" else "stop"
        )

        message_obj: Dict[str, Any] = {"role": "assistant", "content": content_text}
        # OpenWebUI will show this under its Thinking panel.
        # - If this turn contains tool_calls, keep *both* Claude thinking blocks and any plain text
        #   in reasoning_content (so the user doesn't see analysis as normal content).
        # - Otherwise, only the Claude thinking blocks go to reasoning_content.
        if tool_calls:
            rc = "".join(["".join(thinking_parts), "".join(text_parts)])
            if rc:
                message_obj["reasoning_content"] = rc
        else:
            if thinking_parts:
                message_obj["reasoning_content"] = "".join(thinking_parts)

        if tool_calls:
            message_obj["tool_calls"] = tool_calls
        message_obj["_finish_reason"] = finish_reason

        prompt_tokens, completion_tokens = self._extract_usage_tokens(msg)

        # Fallback: estimate tokens with tiktoken if upstream didn't return usage
        if getattr(self.valves, "RETURN_TOKENS", True):
            if prompt_tokens is None:
                prompt_tokens = self._estimate_tokens_from_obj(
                    payload, model_hint=model_for_openai
                )
            if completion_tokens is None:
                # Count what we actually return to OpenWebUI (including <think> wrapper, if present)
                completion_tokens = self._estimate_tokens_from_obj(
                    message_obj, model_hint=model_for_openai
                )
        if getattr(self.valves, "LOG_ENABLED", False):
            logger.info(
                f"Token usage: prompt={prompt_tokens} completion={completion_tokens}"
            )

        return self._openai_response_object(
            message_obj, model_for_openai, prompt_tokens, completion_tokens
        )
