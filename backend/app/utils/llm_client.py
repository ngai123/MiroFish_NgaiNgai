"""
LLM客户端封装
统一使用OpenAI格式调用
"""

import json
import logging
import re
import time
from typing import Optional, Dict, Any, List
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

from ..config import Config

logger = logging.getLogger(__name__)

# Matches all known think-block variants across models
_THINK_RE = re.compile(
    r'<(think|thinking|reason|reasoning)>[\s\S]*?</\1>',
    re.IGNORECASE
)


class LLMClient:
    """LLM客户端"""

    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2  # seconds, doubles each attempt

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """
        发送聊天请求，自动重试（指数退避，最多3次）

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            模型响应文本
        """
        last_exc = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content or ""
                # Remove think/reasoning blocks produced by reasoning models
                content = _THINK_RE.sub('', content).strip()
                return content
            except RateLimitError as exc:
                last_exc = exc
                delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(f"Rate limit hit, retrying in {delay}s (attempt {attempt+1}/{self.MAX_RETRIES})")
                time.sleep(delay)
            except APITimeoutError as exc:
                last_exc = exc
                delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(f"API timeout, retrying in {delay}s (attempt {attempt+1}/{self.MAX_RETRIES})")
                time.sleep(delay)
            except APIError as exc:
                # Non-retryable API errors (4xx) — raise immediately
                raise
        raise last_exc

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON，带重试和健壮解析

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            解析后的JSON对象
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Strip markdown code fences
        cleaned = response.strip()
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned)
        cleaned = cleaned.strip()

        # Extract the outermost JSON object/array even if surrounded by prose
        match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', cleaned)
        if match:
            cleaned = match.group(1)

        try:
            result = json.loads(cleaned)
            # Reject if the model returned an error payload instead of data
            if isinstance(result, dict) and list(result.keys()) == ["error"]:
                raise ValueError(f"LLM返回错误负载: {result}")
            return result
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM返回的JSON格式无效: {cleaned[:500]}") from exc

