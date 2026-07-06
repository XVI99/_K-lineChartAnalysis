# -*- coding: utf-8 -*-
"""
llm_client.py — 大语言模型客户端（火山方舟 Coding Plan / OpenAI 兼容协议）

功能：
- OpenAI 兼容协议调用火山方舟 Ark API
- 自动从环境变量或 config.yaml 读取 API Key
- 失败重试 + 指数退避
- 本地缓存（相同 prompt 不重复调用）
- JSON 模式调用（强制返回合法 JSON）
- 优雅降级（无 API Key 时返回空结果，不崩溃）

用法:
    from AStockQuant.core.llm_client import llm
    resp = llm.chat("分析今天A股市场情绪", temperature=0.3)
    data = llm.chat_json("分析这条新闻情绪，返回JSON", system="你是金融分析师")
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
CACHE_DIR = ROOT / "data_cache" / "llm_cache"


class LLMClient:
    """大语言模型客户端（单例）"""

    _instance: Optional["LLMClient"] = None

    def __new__(cls) -> "LLMClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._config: Dict[str, Any] = {}
        self._client: Optional[OpenAI] = None
        self._api_key: str = ""
        self._enabled: bool = False
        self._cache: Dict[str, float] = {}
        self._load_config()

    def _load_config(self):
        """从 config.yaml 加载 LLM 配置，从 .env 文件加载 API Key"""
        # 先加载 .env 文件（项目根目录）
        try:
            from dotenv import load_dotenv
            env_path = REPO_ROOT / ".env"
            if env_path.exists():
                load_dotenv(env_path, override=False)
        except ImportError:
            pass  # python-dotenv 未安装时回退到纯环境变量

        try:
            import yaml
            cfg_path = ROOT / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    full_cfg = yaml.safe_load(f) or {}
                self._config = full_cfg.get("llm", {})
        except Exception as e:
            print(f"[LLMClient] 配置加载失败: {e}")
            self._config = {}

        self._enabled = self._config.get("enabled", False)

        api_key = self._config.get("api_key", "").strip()
        if not api_key:
            env_name = self._config.get("api_key_env", "ARK_API_KEY")
            api_key = os.environ.get(env_name, "").strip()

        self._api_key = api_key

        if self._enabled and api_key:
            base_url = self._config.get(
                "base_url", "https://ark.cn-beijing.volces.com/api/coding/v3"
            )
            try:
                self._client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=self._config.get("timeout", 60),
                    max_retries=0,
                )
                print(f"[LLMClient] 已初始化, base_url={base_url}")
            except Exception as e:
                print(f"[LLMClient] 初始化失败: {e}")
                self._enabled = False
        else:
            if self._enabled and not api_key:
                print("[LLMClient] LLM 已启用但未找到 API Key，LLM 功能将降级")
                print(f"  请设置环境变量 {self._config.get('api_key_env', 'ARK_API_KEY')} 或在 config.yaml llm.api_key 中填入")
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def _cache_key(self, messages: List[Dict], model: str, **kwargs) -> str:
        raw = json.dumps({"m": messages, "model": model, **kwargs}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _cache_get(self, key: str) -> Optional[str]:
        if not self._config.get("cache_enabled", True):
            return None
        ttl = self._config.get("cache_ttl", 3600)
        path = CACHE_DIR / f"{key}.txt"
        if path.exists():
            mtime = path.stat().st_mtime
            if time.time() - mtime < ttl:
                return path.read_text(encoding="utf-8")
        return None

    def _cache_set(self, key: str, value: str):
        if not self._config.get("cache_enabled", True):
            return
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / f"{key}.txt"
        path.write_text(value, encoding="utf-8")

    def _call_with_retry(
        self,
        messages: List[Dict],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """带重试的 API 调用"""
        max_retries = self._config.get("max_retries", 3)
        retry_delay = self._config.get("retry_delay", 2)

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format:
                    kwargs["response_format"] = response_format
                if timeout is not None:
                    kwargs["timeout"] = timeout

                resp = self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                wait = retry_delay * (2 ** (attempt - 1))
                print(f"[LLMClient] 调用失败 (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    print(f"  {wait}s 后重试...")
                    time.sleep(wait)

        raise RuntimeError(f"LLM 调用失败 ({max_retries} 次): {last_err}")

    def chat(
        self,
        prompt: str,
        system: str = "",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        timeout: Optional[int] = None,
    ) -> str:
        """
        普通对话调用，返回文本

        Args:
            prompt: 用户提示词
            system: 系统提示词（角色设定）
            model: 模型名（留空用默认）
            temperature: 温度（留空用默认）
            max_tokens: 最大输出 token（留空用默认）
            use_cache: 是否使用缓存
            timeout: 请求超时秒（留空用 config 默认 60s）

        Returns:
            模型回复文本（LLM 不可用时返回空字符串）
        """
        if not self._enabled or self._client is None:
            return ""

        model = model or self._config.get("model", "doubao-seed-2.0-pro")
        temperature = temperature if temperature is not None else self._config.get("temperature", 0.3)
        max_tokens = max_tokens or self._config.get("max_tokens", 4096)

        messages: List[Dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if use_cache:
            ck = self._cache_key(messages, model, temperature=temperature, max_tokens=max_tokens)
            cached = self._cache_get(ck)
            if cached is not None:
                return cached

        result = self._call_with_retry(messages, model, temperature, max_tokens, timeout=timeout)

        if use_cache:
            self._cache_set(ck, result)

        return result

    def chat_json(
        self,
        prompt: str,
        system: str = "",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
    ) -> Optional[Dict]:
        """
        JSON 模式调用，强制返回合法 JSON 字典

        Args:
            同 chat()

        Returns:
            解析后的 JSON 字典（失败返回 None）
        """
        if not self._enabled or self._client is None:
            return None

        model = model or self._config.get("model", "doubao-seed-2.0-pro")
        temperature = temperature if temperature is not None else self._config.get("temperature", 0.3)
        max_tokens = max_tokens or self._config.get("max_tokens", 4096)

        messages: List[Dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        json_instruction = f"{prompt}\n\n请严格返回合法 JSON 格式（不要包含 markdown 代码块标记），不要输出任何其他文字。"
        messages.append({"role": "user", "content": json_instruction})

        if use_cache:
            ck = self._cache_key(messages, model, temperature=temperature, max_tokens=max_tokens, json=True)
            cached = self._cache_get(ck)
            if cached is not None:
                try:
                    return json.loads(cached)
                except json.JSONDecodeError:
                    pass

        try:
            result = self._call_with_retry(
                messages, model, temperature, max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            print(f"[LLMClient] chat_json 调用失败: {e}")
            result = self._call_with_retry(messages, model, temperature, max_tokens)

        if use_cache:
            self._cache_set(ck, result)

        try:
            return json.loads(result)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{[\s\S]*\}', result)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            print(f"[LLMClient] JSON 解析失败, 原始输出: {result[:200]}")
            return None

    def batch_chat(
        self,
        prompts: List[str],
        system: str = "",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """批量调用（顺序执行，带缓存）"""
        results = []
        for i, p in enumerate(prompts):
            r = self.chat(p, system=system, model=model, temperature=temperature, max_tokens=max_tokens)
            results.append(r)
            if (i + 1) % 10 == 0:
                print(f"[LLMClient] batch {i+1}/{len(prompts)}")
        return results


llm = LLMClient()
