"""litellm proxy pre-call hook: strip reasoning/thinking from the assistant history."""
from litellm.integrations.custom_logger import CustomLogger

def _clean_messages(msgs):
    if not isinstance(msgs, list):
        return
    for m in msgs:
        if not isinstance(m, dict):
            continue
        m.pop("reasoning_content", None)
        m.pop("reasoning", None)
        m.pop("thinking", None)
        c = m.get("content")
        if isinstance(c, list):
            newc = [p for p in c if not (isinstance(p, dict) and p.get("type") in ("thinking", "reasoning", "redacted_thinking"))]
            if not newc:
                newc = [{"type": "text", "text": ""}]
            m["content"] = newc

class StripThinking(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        try:
            _clean_messages(data.get("messages"))
        except Exception as e:
            print(f"[strip_thinking_hook] warn: {e}")
        return data

strip_thinking = StripThinking()
