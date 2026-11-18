"""VLM calls through the Claude Code CLI (`claude -p`), which is authenticated in the container.

There is no API key or anthropic SDK in the container, so images go inline as base64 content blocks over
`--input-format stream-json`, with no tools and a JSON schema for the answer: one model call per query.
"""

import base64
import io
import json
import shutil
import subprocess
import time

import numpy as np

DEFAULT_MODEL = "claude-opus-5-5"


class VLMError(RuntimeError):
    pass


def png_b64(img):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class ClaudeCLI(object):
    def __init__(self, model=DEFAULT_MODEL, system="", timeout=180, retries=2, effort=None):
        self.exe = shutil.which("claude") or "/root/.local/bin/claude"
        self.model, self.system, self.timeout, self.retries, self.effort = model, system, timeout, retries, effort
        self.calls, self.cost_usd, self.latency_s = 0, 0.0, 0.0

    def ask(self, parts, schema):
        """parts: list of str (text) or HxWx3 uint8 arrays (images), in order. Returns the parsed JSON dict."""
        content = []
        for p in parts:
            if isinstance(p, str):
                content.append({"type": "text", "text": p})
            else:
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                            "data": png_b64(p)}})
        msg = json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"
        cmd = [self.exe, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
               "--tools", "", "--model", self.model, "--no-session-persistence", "--setting-sources", "",
               "--strict-mcp-config", "--system-prompt", self.system, "--json-schema", json.dumps(schema)]
        if self.effort:
            cmd += ["--effort", self.effort]
        last = None
        for attempt in range(self.retries + 1):
            t0 = time.time()
            try:
                p = subprocess.run(cmd, input=msg, capture_output=True, text=True, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                last = "timeout"
                continue
            result = None
            for line in p.stdout.splitlines():
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("type") == "result":
                    result = d
            if result is None or result.get("is_error") or not isinstance(result.get("structured_output"), dict):
                last = (p.stderr or "")[-400:] or str(result)[:400]
                time.sleep(2 * (attempt + 1))
                continue
            self.calls += 1
            self.cost_usd += float(result.get("total_cost_usd") or 0.0)
            self.latency_s += time.time() - t0
            return result["structured_output"]
        raise VLMError(f"VLM call failed after {self.retries + 1} attempts: {last}")
