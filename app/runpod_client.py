# app/runpod_client.py
import os, json, requests, logging
from typing import List, Dict, Any

class RunpodClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("RUNPOD_API_KEY", "")
        self.endpoint_id = os.getenv("RUNPOD_COACH_ENDPOINT_ID", "")
        self.model = os.getenv("COACH_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
        self.base = f"https://api.runpod.ai/v2/{self.endpoint_id}/openai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.last_error: str | None = None

    # ---------- simple GET ----------
    def models(self) -> Dict[str, Any]:
        try:
            r = requests.get(f"{self.base}/models", headers=self.headers, timeout=30)
        except Exception as e:
            return {"ok": False, "error": f"request_error:{e.__class__.__name__}:{e}"}
        try:
            return r.json()
        except Exception:
            return {"status": r.status_code, "body": r.text}

    # ---------- POST helpers ----------
    def _post(self, path: str, payload: Dict[str, Any], timeout: int = 120) -> requests.Response:
        return requests.post(f"{self.base}{path}", headers=self.headers, json=payload, timeout=timeout)

    def chat(self, messages: List[Dict[str, str]], **opts) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": self.model, "messages": messages}
        payload.update({k: v for k, v in opts.items() if v is not None})
        try:
            r = self._post("/chat/completions", payload)
        except Exception as e:
            self.last_error = f"chat_request_error:{e}"
            logging.exception("RunPod chat request error")
            return {"ok": False, "status": 0, "error": str(e)}
        if r.status_code != 200:
            self.last_error = f"chat_http_{r.status_code}:{r.text[:400]}"
            return {"ok": False, "status": r.status_code, "error": r.text}
        try:
            return {"ok": True, **r.json()}
        except Exception:
            return {"ok": True, "raw": r.text}

    def completions(self, prompt: str, **opts) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": self.model, "prompt": prompt}
        payload.update({k: v for k, v in opts.items() if v is not None})
        try:
            r = self._post("/completions", payload)
        except Exception as e:
            self.last_error = f"completions_request_error:{e}"
            logging.exception("RunPod completions request error")
            return {"ok": False, "status": 0, "error": str(e)}
        if r.status_code != 200:
            self.last_error = f"completions_http_{r.status_code}:{r.text[:400]}"
            return {"ok": False, "status": r.status_code, "error": r.text}
        try:
            return {"ok": True, **r.json()}
        except Exception:
            return {"ok": True, "raw": r.text}

    # ---------- chat→completion fallback ----------
    @staticmethod
    def to_prompt(messages: List[Dict[str, str]]) -> str:
        parts: list[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"<<SYS>>\n{content}\n<</SYS>>\n")
            else:
                parts.append(f"{role.upper()}: {content}\n")
        parts.append("ASSISTANT:")
        return "\n".join(parts)

    def chat_fallback(self, messages: List[Dict[str, str]], **opts) -> Dict[str, Any]:
        # 1) try chat
        r = self.chat(messages, **opts)
        if r.get("ok"):
            content = ((r.get("choices") or [{}])[0].get("message") or {}).get("content")
            return {"ok": True, "mode": "chat", "status": 200, "content": content, "raw": r}
        # 2) fall back to completions
        prompt = self.to_prompt(messages)
        r2 = self.completions(prompt, **opts)
        if r2.get("ok"):
            text = ((r2.get("choices") or [{}])[0].get("text")) or ""
            return {"ok": True, "mode": "completion", "status": 200, "content": text, "raw": r2, "prev_error": r}
        return {
            "ok": False,
            "status": r2.get("status", r.get("status")),
            "error": r2.get("error") or r.get("error"),
            "prev_error": r,
        }

client = RunpodClient()