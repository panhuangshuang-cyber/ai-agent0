import os
import json
import logging
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# We will let the user override this with an env variable, falling back to a default list.
CANDIDATE_MODELS = [m.strip() for m in os.getenv("GLM_MODELS", "deepseek-v4.1-flash,glm-5.3-flash,mimo-v2.6-flash,glm-5-turbo,glm-5.3,kimi-k3,minimax-m3,mimo-v2.6-pro,glm-5.1").split(",") if m.strip()]
_DISABLED_MODELS = set()


async def chat_with_fallback(messages, tools=None, tool_choice=None):
    client = AsyncOpenAI(
        api_key=os.getenv("TENCENT_CLOUD_API_KEY"),
        base_url=os.getenv("TENCENT_CLOUD_API_BASE", "https://tokenhub.tencentmaas.com/v1")
    )
    
    last_exception = None
    try:
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
                    
                response = await client.chat.completions.create(**kwargs)
                return response, model
            except Exception as e:
                err_str = str(e)
                logging.warning(f"Model {model} failed: {e}. Switching to next model.")
                if "402" in err_str or "400004" in err_str or "401008" in err_str:
                    logging.error(f"Permanently disabling unavailable model: {model}")
                    _DISABLED_MODELS.add(model)
                last_exception = e
                
        raise RuntimeError(f"All models in the candidate list failed. Last error: {last_exception}")
    finally:
        await client.close()

