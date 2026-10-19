#!/usr/bin/env python3
"""toolcall_shim.py — OpenAI-compatible shim between the litellm proxy and the SGLang servers for the
H-step proposer: turns <tool_call> text in the model output into OpenAI tool_calls.

Run: python3 toolcall_shim.py   (listens on SHIM_LISTEN_BASE+i, forwards to SHIM_UP_BASE+i, i in 0..SHIM_N-1)
Env vars:
  SHIM_N            number of servers (default 8)
  SHIM_LISTEN_BASE  first listen port (default 8100)
  SHIM_UP_BASE      first SGLang server port (default 8000)
  SHIM_LOG          log file (default $SELFHPO_WORK/logs/toolcall_shim.log)
"""
import asyncio, json, os, re, time, uuid
from aiohttp import web, ClientSession, ClientTimeout

N = int(os.environ.get("SHIM_N", "8"))
LISTEN_BASE = int(os.environ.get("SHIM_LISTEN_BASE", "8100"))
UP_BASE = int(os.environ.get("SHIM_UP_BASE", "8000"))
LOG = os.environ.get("SHIM_LOG", os.path.join(os.environ.get("SELFHPO_WORK", "."), "logs", "toolcall_shim.log"))

TC_RE = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*(?:</tool_call>)?", re.S)
P_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def _val(v):
    s = v.strip()
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if s in ("true", "false"):
        return s == "true"
    if s[:1] in "[{":
        try:
            return json.loads(s)
        except Exception:
            pass
    return v


def extract(text):
    if not text or "<tool_call>" not in text:
        return [], text
    calls = []
    for m in TC_RE.finditer(text):
        args = {k: _val(v) for k, v in P_RE.findall(m.group(2))}
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": m.group(1), "arguments": json.dumps(args)}})
    if not calls:
        return [], text
    cut = text.find("<tool_call>")
    return calls, text[:cut].rstrip()


def repair(resp, port):
    try:
        ch = resp["choices"][0]
        msg = ch["message"]
    except Exception:
        return resp
    if msg.get("tool_calls"):
        return resp
    for field in ("reasoning_content", "content"):
        calls, rest = extract(msg.get(field) or "")
        if calls:
            msg[field] = rest
            msg["tool_calls"] = calls
            ch["finish_reason"] = "tool_calls"
            try:
                with open(LOG, "a") as f:
                    f.write(f"{time.strftime('%FT%TZ', time.gmtime())} port={port} field={field} "
                            f"repaired={[c['function']['name'] for c in calls]}\n")
            except Exception:
                pass
            break
    return resp


def sse(resp):
    ch = resp["choices"][0]
    msg = ch["message"]
    base = {"id": resp.get("id", "chatcmpl-shim"), "object": "chat.completion.chunk",
            "created": resp.get("created", int(time.time())), "model": resp.get("model", "")}
    delta = {"role": "assistant"}
    if msg.get("reasoning_content"):
        delta["reasoning_content"] = msg["reasoning_content"]
    if msg.get("content"):
        delta["content"] = msg["content"]
    out = [dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": None}])]
    for i, tc in enumerate(msg.get("tool_calls") or []):
        out.append(dict(base, choices=[{"index": 0, "delta": {"tool_calls": [dict(tc, index=i)]},
                                        "finish_reason": None}]))
    last = dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": ch.get("finish_reason") or "stop"}])
    if resp.get("usage"):
        last["usage"] = resp["usage"]
    out.append(last)
    return "".join(f"data: {json.dumps(c)}\n\n" for c in out) + "data: [DONE]\n\n"


def make_app(i, session):
    up = f"http://127.0.0.1:{UP_BASE + i}"

    async def chat(request):
        body = await request.json()
        stream = bool(body.pop("stream", False))
        body.pop("stream_options", None)
        async with session.post(up + "/v1/chat/completions", json=body) as r:
            txt = await r.text()
            if r.status != 200:
                return web.Response(status=r.status, text=txt, content_type="application/json")
        resp = repair(json.loads(txt), UP_BASE + i)
        if not stream:
            return web.json_response(resp)
        return web.Response(text=sse(resp), content_type="text/event-stream")

    async def passthru(request):
        async with session.request(request.method, up + request.path_qs,
                                   data=await request.read()) as r:
            return web.Response(status=r.status, body=await r.read(),
                                content_type=r.content_type)

    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_route("*", "/{tail:.*}", passthru)
    return app


async def main():
    session = ClientSession(timeout=ClientTimeout(total=1800))
    runners = []
    for i in range(N):
        runner = web.AppRunner(make_app(i, session))
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", LISTEN_BASE + i).start()
        runners.append(runner)
    print(f"SHIM_READY {LISTEN_BASE}..{LISTEN_BASE + N - 1} -> {UP_BASE}..", flush=True)
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
