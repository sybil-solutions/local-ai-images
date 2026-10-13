#!/usr/bin/env python3
"""Gateway tests against a fake engine. Standard library only; run: python3 test.py

The fake engine speaks OpenAI chat completions, streaming and not, and
answers with a tool call whenever the request carries tools and the last
user message asks for one. Each test asserts the exact shapes Claude Code
(Anthropic Messages) and Codex (OpenAI Responses) rely on.
"""

import json
import os
import select
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_PORT, GATEWAY_PORT = 18000, 18434
KEY = "test-key-123"
DEADLINE, SLOTS = 1.0, 4
# A second engine+gateway pair for the cancel-on-disconnect tests: the engine answers slowly and
# records when its client (the gateway) closes, so we can see the gateway cut it off.
SLOW_ENGINE_PORT, SLOW_GATEWAY_PORT = 18001, 18435
GATEWAY_FILE = os.environ.get("GATEWAY_FILE", os.path.join(HERE, "gateway.py"))


class FakeEngine(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    last_request = None

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"object": "list", "data": [{"id": "fake-model", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(length))
        FakeEngine.last_request = req
        if req.get("test_status") in (400, 429, 500):
            body = json.dumps({"error": {"message": "Unsupported reasoning effort", "type": "BadRequestError"}}).encode()
            self.send_response(req["test_status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        last_user = next((m for m in reversed(req["messages"]) if m["role"] == "user"), {"content": ""})
        wants_tool = bool(req.get("tools")) and "use the tool" in str(last_user.get("content", ""))
        has_tool_result = any(m["role"] == "tool" for m in req["messages"])
        if wants_tool and not has_tool_result:
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_abc", "type": "function", "function": {"name": "shell", "arguments": json.dumps({"command": "echo hi"})}}]}
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": "Hello from the engine"}
            finish = "stop"
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            def sse(obj):
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode()); self.wfile.flush()
            base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": req["model"]}
            if message.get("tool_calls"):
                call = message["tool_calls"][0]
                sse(dict(base, choices=[{"index": 0, "delta": {"role": "assistant", "tool_calls": [{"index": 0, "id": call["id"], "type": "function", "function": {"name": call["function"]["name"], "arguments": ""}}]}, "finish_reason": None}]))
                for piece in ('{"command": ', '"echo hi"}'):
                    sse(dict(base, choices=[{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": piece}}]}, "finish_reason": None}]))
            else:
                for word in ("Hello ", "from ", "the ", "engine"):
                    sse(dict(base, choices=[{"index": 0, "delta": {"content": word}, "finish_reason": None}]))
            sse(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": finish}]))
            sse(dict(base, choices=[], usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}))
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
            self.close_connection = True
            return
        body = json.dumps({"id": "chatcmpl-1", "object": "chat.completion", "model": req["model"],
                           "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                           "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class SlowEngine(BaseHTTPRequestHandler):
    """A deliberately slow engine that records when its client (the gateway) closes the connection.

    Non-streamed: it holds the response headers for ANSWER_DELAY seconds (a long decode), peeking its
    own socket for EOF the whole time. Streamed: it sends the first chunk after one STEP, then a chunk
    every STEP, peeking between them. The moment the gateway shuts the upstream connection, the peek
    returns empty and the time is recorded in `seen`, which is how a cancel-on-disconnect shows up.
    """
    protocol_version = "HTTP/1.1"
    ANSWER_DELAY = 2.5          # a non-streamed answer is withheld this long
    STEP = 0.4                  # gap between streamed chunks
    WORDS = ("Hello ", "from ", "the ", "engine")
    lock = threading.Lock()
    seen = []                   # one record per request: {"path", "stream", "start", "disconnect", "completed"}

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"object": "list", "data": [{"id": "fake-model", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def client_gone(self):
        try:
            ready, _, _ = select.select([self.connection], [], [], 0)
        except OSError:
            return True
        if not ready:
            return False
        try:
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

    def wait(self, record, seconds):
        """Sleep up to `seconds`, returning True if the client disconnected first (recording when)."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.client_gone():
                with SlowEngine.lock:
                    record["disconnect"] = time.monotonic() - record["start"]
                return True
            time.sleep(0.05)
        return False

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(length) or b"{}")
        record = {"path": self.path, "stream": bool(req.get("stream")), "start": time.monotonic(), "disconnect": None, "completed": False}
        with SlowEngine.lock:
            SlowEngine.seen.append(record)
        base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": req.get("model", "fake-model")}
        try:
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for word in self.WORDS:
                    if self.wait(record, self.STEP):
                        return
                    self.wfile.write(f"data: {json.dumps(dict(base, choices=[{'index': 0, 'delta': {'content': word}, 'finish_reason': None}]))}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(f"data: {json.dumps(dict(base, choices=[{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]))}\n\n".encode())
                self.wfile.write(f"data: {json.dumps(dict(base, choices=[], usage={'prompt_tokens': 10, 'completion_tokens': 4, 'total_tokens': 14}))}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True
            else:
                if self.wait(record, self.ANSWER_DELAY):  # withhold the whole answer, like a long non-streamed decode
                    return
                body = json.dumps({"id": "chatcmpl-1", "object": "chat.completion", "model": req.get("model", "fake-model"),
                                   "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello from the engine"}, "finish_reason": "stop"}],
                                   "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            with SlowEngine.lock:
                record["completed"] = True
        except OSError:  # the gateway shut the connection mid-answer
            if record["disconnect"] is None:
                with SlowEngine.lock:
                    record["disconnect"] = time.monotonic() - record["start"]


def raw_post(port, path, body, disconnect_after=None):
    """Send one POST over a raw socket. If disconnect_after is set, close the socket then; else read the full reply."""
    data = json.dumps(body).encode()
    head = (f"POST {path} HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {KEY}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(data)}\r\nConnection: close\r\n\r\n").encode()
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    sock.sendall(head + data)
    if disconnect_after is not None:
        time.sleep(disconnect_after)
        sock.close()
        return None
    sock.settimeout(10)
    chunks = []
    while True:
        try:
            piece = sock.recv(65536)
        except socket.timeout:
            break
        if not piece:
            break
        chunks.append(piece)
    sock.close()
    return b"".join(chunks)


def call(path, body=None, key=KEY, stream=False):
    req = urllib.request.Request(f"http://127.0.0.1:{GATEWAY_PORT}{path}", data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            if stream:
                events = []
                for line in response:
                    line = line.decode().strip()
                    if line.startswith("data:") and line[5:].strip() != "[DONE]":
                        events.append(json.loads(line[5:].strip()))
                return response.status, events
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


passed = 0


def check(name, condition, detail=""):
    global passed
    if condition:
        passed += 1
        print(f"ok - {name}")
    else:
        print(f"not ok - {name} {detail}", file=sys.stderr)
        sys.exit(1)


def disconnect_tests(key_file):
    """A client that drops mid-answer must make the gateway cut the engine off on every proxied path."""
    engine = ThreadingHTTPServer(("127.0.0.1", SLOW_ENGINE_PORT), SlowEngine)
    threading.Thread(target=engine.serve_forever, daemon=True).start()
    env = dict(os.environ, UPSTREAM=f"http://127.0.0.1:{SLOW_ENGINE_PORT}", GATEWAY_PORT=str(SLOW_GATEWAY_PORT),
               GATEWAY_KEY_FILE=key_file, MODEL="fake-model")
    gateway = subprocess.Popen([sys.executable, GATEWAY_FILE], env=env, stderr=subprocess.PIPE)
    # These paths each reach the engine through a different gateway code path (chat passthrough,
    # /v1/completions passthrough, Anthropic Messages, OpenAI Responses), streamed and not.
    cases = [
        ("chat non-stream", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]}),
        ("chat stream", "/v1/chat/completions", {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}),
        ("completions non-stream", "/v1/completions", {"model": "m", "prompt": "hi"}),
        ("messages non-stream", "/v1/messages", {"model": "m", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}),
        ("messages stream", "/v1/messages", {"model": "m", "max_tokens": 64, "stream": True, "messages": [{"role": "user", "content": "hi"}]}),
        ("responses non-stream", "/v1/responses", {"model": "m", "input": "hi"}),
        ("responses stream", "/v1/responses", {"model": "m", "stream": True, "input": "hi"}),
    ]
    try:
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{SLOW_GATEWAY_PORT}/health", timeout=1); break
            except Exception:
                time.sleep(0.1)
        for label, path, body in cases:
            # A client that disconnects after ~1s: the engine must observe its connection closed within ~2s.
            before = len(SlowEngine.seen)
            raw_post(SLOW_GATEWAY_PORT, path, body, disconnect_after=1.0)
            record = None
            for _ in range(40):  # wait up to 2s past the disconnect for the engine to notice
                with SlowEngine.lock:
                    record = SlowEngine.seen[before] if len(SlowEngine.seen) > before else None
                    noticed = record is not None and record["disconnect"] is not None
                if noticed:
                    break
                time.sleep(0.05)
            check(f"disconnect cancels the engine: {label}",
                  record is not None and record["disconnect"] is not None and record["disconnect"] < 3.0 and not record["completed"],
                  str(record))
            # A client that stays connected still gets the full answer on the same path. A streamed answer
            # arrives split across SSE deltas, so assert every word is present rather than one contiguous string.
            reply = raw_post(SLOW_GATEWAY_PORT, path, body)
            check(f"a connected client still gets the full answer: {label}",
                  reply is not None and all(word.strip().encode() in reply for word in SlowEngine.WORDS),
                  (reply or b"")[-200:].decode(errors="replace"))
    finally:
        gateway.terminate()
        engine.shutdown()


def main():
    engine = ThreadingHTTPServer(("127.0.0.1", ENGINE_PORT), FakeEngine)
    threading.Thread(target=engine.serve_forever, daemon=True).start()
    key_file = tempfile.NamedTemporaryFile("w", delete=False)
    key_file.write(KEY + "\n"); key_file.close()
    usage_log = os.path.join(tempfile.mkdtemp(), "usage.jsonl")
    env = dict(os.environ, UPSTREAM=f"http://127.0.0.1:{ENGINE_PORT}", GATEWAY_PORT=str(GATEWAY_PORT), GATEWAY_KEY_FILE=key_file.name, MODEL="fake-model",
               GATEWAY_USAGE_LOG=usage_log, GATEWAY_REQUEST_DEADLINE=str(DEADLINE), GATEWAY_MAX_CONNECTIONS=str(SLOTS))
    gateway = subprocess.Popen([sys.executable, GATEWAY_FILE], env=env, stderr=subprocess.PIPE)
    try:
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{GATEWAY_PORT}/health", timeout=1); break
            except Exception:
                time.sleep(0.1)
        status, body = call("/v1/models")
        check("models passthrough", status == 200 and body["data"][0]["id"] == "fake-model", str(body))
        status, body = call("/v1/models", key="")
        check("missing key is 401", status == 401, str(status))
        status, body = call("/v1/chat/completions", {"model": "anything", "messages": [{"role": "user", "content": "hi"}]})
        check("chat passthrough maps the model alias", status == 200 and FakeEngine.last_request["model"] == "fake-model", str(body))
        for path in ("/v1/chat/completions", "/v1/completions"):
            for expected in (400, 429, 500):
                status, body = call(path, {"model": "anything", "stream": True, "test_status": expected,
                                          "messages": [{"role": "user", "content": "hi"}]}, stream=True)
                check(f"{path}: streaming preserves upstream {expected}",
                      status == expected and body["error"]["message"] == "Unsupported reasoning effort", str(body))

        # Anthropic Messages, non-streaming
        status, body = call("/v1/messages", {"model": "claude-x", "max_tokens": 64, "system": "be brief", "messages": [{"role": "user", "content": "hi"}]})
        check("messages: text reply", status == 200 and body["type"] == "message" and body["content"][0]["text"] == "Hello from the engine" and body["stop_reason"] == "end_turn", str(body))
        check("messages: system becomes a system message", FakeEngine.last_request["messages"][0] == {"role": "system", "content": "be brief"}, str(FakeEngine.last_request))
        tools = [{"name": "shell", "description": "run", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}}]
        status, body = call("/v1/messages", {"model": "claude-x", "max_tokens": 64, "tools": tools, "messages": [{"role": "user", "content": "please use the tool"}]})
        check("messages: tool_use block", status == 200 and body["stop_reason"] == "tool_use" and body["content"][0]["type"] == "tool_use" and body["content"][0]["input"] == {"command": "echo hi"}, str(body))
        tool_id = body["content"][0]["id"]
        status, body = call("/v1/messages", {"model": "claude-x", "max_tokens": 64, "tools": tools, "messages": [
            {"role": "user", "content": "please use the tool"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": tool_id, "name": "shell", "input": {"command": "echo hi"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "hi"}]}]})
        check("messages: tool_result round trip", status == 200 and body["content"][0]["type"] == "text", str(body))
        chat_msgs = FakeEngine.last_request["messages"]
        check("messages: tool_result became a tool message", any(m["role"] == "tool" and m["tool_call_id"] == tool_id for m in chat_msgs), str(chat_msgs))
        status, events = call("/v1/messages", {"model": "claude-x", "max_tokens": 64, "stream": True, "messages": [{"role": "user", "content": "hi"}]}, stream=True)
        kinds = [e["type"] for e in events]
        check("messages: stream event order", kinds[0] == "message_start" and "content_block_start" in kinds and kinds[-2] == "message_delta" and kinds[-1] == "message_stop", str(kinds))
        text = "".join(e["delta"]["text"] for e in events if e["type"] == "content_block_delta")
        check("messages: streamed text", text == "Hello from the engine", text)
        status, events = call("/v1/messages", {"model": "claude-x", "max_tokens": 64, "stream": True, "tools": tools, "messages": [{"role": "user", "content": "please use the tool"}]}, stream=True)
        starts = [e for e in events if e["type"] == "content_block_start"]
        partial = "".join(e["delta"]["partial_json"] for e in events if e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta")
        check("messages: streamed tool_use", starts and starts[0]["content_block"]["type"] == "tool_use" and json.loads(partial) == {"command": "echo hi"}, str(events))
        check("messages: streamed stop_reason tool_use", any(e["type"] == "message_delta" and e["delta"]["stop_reason"] == "tool_use" for e in events), str(events))

        # OpenAI Responses, non-streaming
        status, body = call("/v1/messages", {"model": "claude-x", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "edit", "description": "", "input_schema": {"type": "object", "properties": {"path": {"type": "string", "pattern": "^[A-Za-z0-9_\\-.]+$", "format": "uri"}}, "required": ["path"]}}]})
        params = FakeEngine.last_request["tools"][0]["function"]["parameters"]
        check("tools: pattern and format are scrubbed from parameter schemas", "pattern" not in params["properties"]["path"] and "format" not in params["properties"]["path"] and params["required"] == ["path"], str(params))
        status, body = call("/v1/responses", {"model": "gpt-x", "instructions": "be brief", "input": "hi"})
        check("responses: text output", status == 200 and body["object"] == "response" and body["status"] == "completed" and body["output"][0]["content"][0]["text"] == "Hello from the engine", str(body))
        rtools = [{"type": "function", "name": "shell", "description": "run", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}]
        status, body = call("/v1/responses", {"model": "gpt-x", "instructions": "be brief",
            "input": [{"type": "message", "role": "user", "content": "hi"}, {"type": "message", "role": "developer", "content": "answer in French"}]})
        sys_msgs = [m for m in FakeEngine.last_request["messages"] if m["role"] == "system"]
        check("responses: every system/developer message is hoisted into one leading system message",
              len(sys_msgs) == 1 and FakeEngine.last_request["messages"][0]["role"] == "system" and "French" in sys_msgs[0]["content"], str(FakeEngine.last_request["messages"]))
        status, body = call("/v1/chat/completions", {"model": "anything", "messages": [{"role": "user", "content": "hi"}, {"role": "system", "content": "late system"}]})
        check("chat: a mid-list system message is hoisted to the front", FakeEngine.last_request["messages"][0] == {"role": "system", "content": "late system"}, str(FakeEngine.last_request["messages"]))
        status, body = call("/v1/responses", {"model": "gpt-x", "tools": rtools, "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "please use the tool"}]}]})
        fc = [o for o in body["output"] if o["type"] == "function_call"]
        check("responses: function_call item", status == 200 and fc and fc[0]["name"] == "shell" and json.loads(fc[0]["arguments"]) == {"command": "echo hi"}, str(body))
        status, body = call("/v1/responses", {"model": "gpt-x", "tools": rtools, "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "please use the tool"}]},
            {"type": "function_call", "call_id": fc[0]["call_id"], "name": "shell", "arguments": fc[0]["arguments"]},
            {"type": "function_call_output", "call_id": fc[0]["call_id"], "output": "hi"}]})
        check("responses: function_call_output round trip", status == 200 and body["output"][0]["type"] == "message", str(body))
        status, events = call("/v1/responses", {"model": "gpt-x", "stream": True, "input": "hi"}, stream=True)
        kinds = [e["type"] for e in events]
        check("responses: stream event order", kinds[:2] == ["response.created", "response.in_progress"] and "response.output_item.added" in kinds and "response.output_text.done" in kinds and kinds[-1] == "response.completed", str(kinds))
        check("responses: stream sequence numbers", [e["sequence_number"] for e in events] == list(range(1, len(events) + 1)), str([e["sequence_number"] for e in events]))
        final = events[-1]["response"]
        check("responses: completed carries the output", final["status"] == "completed" and final["output"][0]["content"][0]["text"] == "Hello from the engine", str(final))
        status, events = call("/v1/responses", {"model": "gpt-x", "stream": True, "tools": rtools, "input": "please use the tool"}, stream=True)
        done = [e for e in events if e["type"] == "response.function_call_arguments.done"]
        check("responses: streamed function call", done and json.loads(done[0]["arguments"]) == {"command": "echo hi"} and events[-1]["response"]["output"][0]["type"] == "function_call", str(events))
        # Image bytes must survive both dialects, including Read tool results.
        image_url = "data:image/png;base64,aW1hZ2U="
        aimage = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="}}
        expected = [{"type": "text", "text": "describe"}, {"type": "image_url", "image_url": {"url": image_url}}]
        for stream in (False, True):
            status, _ = call("/v1/messages", {"model": "test", "max_tokens": 64, "stream": stream,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "describe"}, aimage]}]}, stream=stream)
            check(f"messages: image reaches engine (stream={stream})", status == 200 and FakeEngine.last_request["messages"][0]["content"] == expected)
            status, _ = call("/v1/responses", {"model": "test", "stream": stream,
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "describe"}, {"type": "input_image", "image_url": image_url}]}]}, stream=stream)
            check(f"responses: image reaches engine (stream={stream})", status == 200 and FakeEngine.last_request["messages"][0]["content"] == expected)
        status, _ = call("/v1/messages", {"model": "test", "messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "read_image", "name": "read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "read_image", "content": [aimage]}]}]})
        messages = FakeEngine.last_request["messages"]
        check("messages: Read image survives tool result", status == 200 and messages[-2]["role"] == "tool" and messages[-2]["tool_call_id"] == "read_image" and messages[-1]["content"] == expected[1:])
        for path, body in [
            ("/v1/messages", {"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "https://example.com/image.png"}}]}]}),
            ("/v1/responses", {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "https://example.com/image.png", "detail": "low"}]}]})]:
            status, _ = call(path, dict(body, model="test"))
            image = FakeEngine.last_request["messages"][0]["content"][0]["image_url"]
            check(f"{path}: image URL preserved", status == 200 and image["url"] == "https://example.com/image.png" and (path.endswith("messages") or image["detail"] == "low"))
        status, body = call("/v1/responses", {"model": "test", "input": [{"role": "user", "content": [{"type": "input_image", "file_id": "unavailable"}]}]})
        check("unsupported image reference fails explicitly", status == 400 and body["error"]["type"] == "invalid_request_error")
        # streamed chat: the engine is asked for usage, the client only sees it when it asked
        status, events = call("/v1/chat/completions", {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}, stream=True)
        check("stream: engine is asked for usage", FakeEngine.last_request.get("stream_options") == {"include_usage": True}, str(FakeEngine.last_request.get("stream_options")))
        check("stream: the usage chunk is hidden from a client that did not ask", status == 200 and not any(e.get("usage") and not e.get("choices") for e in events) and any(e.get("choices") for e in events), str(events[-2:]))
        status, events = call("/v1/chat/completions", {"model": "m", "stream": True, "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": "hi"}]}, stream=True)
        check("stream: a client that asked for usage gets it", any(e.get("usage") and not e.get("choices") for e in events), str(events[-2:]))
        # usage log: one line per answered request, tokens from the engine's usage block
        time.sleep(0.2)
        rows = [json.loads(line) for line in open(usage_log)]
        apis = {row["api"] for row in rows}
        check("usage log: every dialect is recorded", apis == {"chat", "messages", "responses"}, str(apis))
        check("usage log: engine usage is used verbatim, streamed chat included", all(row["completion"] == 4 and row["prompt"] == 10 and not row["estimated"] for row in rows), str([r for r in rows if r["estimated"] or r["prompt"] != 10][:3]))
        check("usage log: model, timing and time-to-first-token are present", all(row["model"] == "fake-model" and row["ms"] >= row["ttft_ms"] >= 0 for row in rows), str(rows[:1]))

        # A client without the key cannot hold the gateway: a request trickled in byte by byte is cut at the deadline,
        # and with every slot held an extra connection is closed, until the deadline frees the slots again.
        slow = socket.create_connection(("127.0.0.1", GATEWAY_PORT))
        slow.settimeout(DEADLINE + 3)
        t0, closed = time.monotonic(), False
        try:
            for byte in b"GET /v1/models HTTP/1.1\r\nX-Slow: " + b"a" * 100:
                slow.sendall(bytes([byte]))
                time.sleep(0.05)
        except OSError:
            closed = True
        if not closed:
            try:
                closed = slow.recv(1) == b""
            except TimeoutError:
                pass
        slow.close()
        check("a request trickled in is cut at the deadline", closed and time.monotonic() - t0 < DEADLINE + 1.5, f"{time.monotonic() - t0:.1f}s")
        idle = [socket.create_connection(("127.0.0.1", GATEWAY_PORT)) for _ in range(SLOTS)]
        time.sleep(0.2)
        extra = socket.create_connection(("127.0.0.1", GATEWAY_PORT))
        extra.settimeout(2)
        try:
            refused = extra.recv(1) == b""
        except OSError:
            refused = True
        extra.close()
        check("with every slot held, one more connection is closed", refused)
        time.sleep(DEADLINE + 0.5)
        status, body = call("/v1/models")
        check("idle connections are dropped at the deadline and the gateway answers again", status == 200, str(status))
        for sock in idle:
            sock.close()
        # Health requests pipelined on every slot, their answers never read: without the key a connection gets one
        # answer and is closed, so the slots are free again within the deadline.
        piped = []
        for _ in range(SLOTS):
            sock = socket.socket()
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
            sock.connect(("127.0.0.1", GATEWAY_PORT))
            sock.settimeout(2)
            try:
                sock.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n" * 2000)
            except OSError:
                pass
            piped.append(sock)
        time.sleep(DEADLINE + 0.5)
        status, body = call("/v1/models")
        check("pipelined health requests with unread answers do not hold the slots", status == 200, str(status))
        for sock in piped:
            sock.close()
        # cancel-on-disconnect: every proxied path, streamed and not, against a slow engine
        disconnect_tests(key_file.name)
        print(f"# {passed} passed")
    finally:
        gateway.terminate()
        engine.shutdown()
        os.unlink(key_file.name)


if __name__ == "__main__":
    main()
