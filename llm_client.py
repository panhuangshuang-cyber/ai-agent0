import asyncio
import os
import json
import logging
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# We will let the user override this with an env variable, falling back to a default list.
CANDIDATE_MODELS = [m.strip() for m in os.getenv("GLM_MODELS", "deepseek-v4.1-flash,glm-5.3-flash,mimo-v2.6-flash,glm-5-turbo,glm-5.3,kimi-k3,minimax-m3,mimo-v2.6-pro,glm-5.1").split(",") if m.strip()]
_DISABLED_MODELS = set()


def _llm_timeout() -> float:
    try:
        return float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    except (TypeError, ValueError):
        return 60.0


def _llm_total_timeout() -> float:
    try:
        return float(os.getenv("LLM_TOTAL_TIMEOUT_SECONDS", "120"))
    except (TypeError, ValueError):
        return 120.0


async def chat_with_fallback(messages, tools=None, tool_choice=None):
    timeout = _llm_timeout()
    total_timeout = _llm_total_timeout()
    client = AsyncOpenAI(
        api_key=os.getenv("TENCENT_CLOUD_API_KEY"),
        base_url=os.getenv("TENCENT_CLOUD_API_BASE", "https://tokenhub.tencentmaas.com/v1"),
        timeout=timeout,
    )

    last_exception = None
    try:
        async def _loop():
            nonlocal last_exception
            for model in CANDIDATE_MODELS:
                if model in _DISABLED_MODELS:
                    continue
                try:
                    logging.info(f"Trying model: {model}")
                    kwargs = {"model": model, "messages": messages}
                    if tools:
                        kwargs["tools"] = tools
                    if tool_choice:
                        kwargs["tool_choice"] = tool_choice

                    try:
                        response = await asyncio.wait_for(
                            client.chat.completions.create(**kwargs), timeout
                        )
                    except (asyncio.TimeoutError, TimeoutError) as exc:
                        # 超时只换下一个模型，不永久禁用。
                        logging.warning(f"Model {model} timed out after {timeout}s. Switching.")
                        last_exception = exc
                        continue
                    return response, model
                except Exception as e:
                    err_str = str(e)
                    logging.warning(f"Model {model} failed: {e}. Switching to next model.")
                    if "402" in err_str or "400004" in err_str or "401008" in err_str:
                        logging.error(f"Permanently disabling unavailable model: {model}")
                        _DISABLED_MODELS.add(model)
                    last_exception = e

            raise RuntimeError(f"All models in the candidate list failed. Last error: {last_exception}")

        try:
            return await asyncio.wait_for(_loop(), total_timeout)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            logging.warning("LLM fallback loop timed out after %ss", total_timeout)
            raise TimeoutError(f"LLM 调用超时（超过 {int(total_timeout)} 秒）") from exc
    finally:
        await client.close()

