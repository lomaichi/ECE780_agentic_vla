"""VLM backends for the speed test: the Claude CLI (fa_core.vlm.ClaudeCLI), the Moondream cloud API and
Moondream2 on the local GPU (moondream_server.py).

MoondreamAPI / MoondreamLocal have the same interface as ClaudeCLI (ask(parts, schema) -> dict), so judge_agent.py
works with any of them. Cloud API key: $MOONDREAM_API_KEY or /home/DockerShared/.moondream_api_key (host: ~/DockerShared/...).
"""

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "failure_agent_prototype"))
from fa_core.vlm import DEFAULT_MODEL, ClaudeCLI, VLMError, png_b64  # noqa: E402,F401


# ------------------------------------------------------------------ Moondream (cloud API) -----------

MOONDREAM_DEFAULT = "moondream3.1-9B-A2B"
MOONDREAM_URL = "https://api.moondream.ai/v1/query"
MOONDREAM_KEY_FILE = "/home/DockerShared/.moondream_api_key"   # host: ~/DockerShared/.moondream_api_key


def _moondream_key():
    import os

    key = os.environ.get("MOONDREAM_API_KEY", "").strip()
    if not key and os.path.exists(MOONDREAM_KEY_FILE):
        key = open(MOONDREAM_KEY_FILE).read().strip()
    if not key:
        raise VLMError(f"no Moondream API key: set MOONDREAM_API_KEY or put it in {MOONDREAM_KEY_FILE}")
    return key


def _mosaic(images, labels):
    """One image from several (the /query endpoint takes one): each image on its own row, labelled."""
    from PIL import Image, ImageDraw

    rows, W = [], 0
    for img, lab in zip(images, labels):
        im = Image.fromarray(np.asarray(img, dtype=np.uint8))
        if im.width > 1024:
            im = im.resize((1024, int(im.height * 1024 / im.width)))
        rows.append((im, lab))
        W = max(W, im.width)
    H = sum(im.height + 24 for im, _ in rows)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    g, y = ImageDraw.Draw(canvas), 0
    for im, lab in rows:
        g.text((6, y + 5), lab, fill=(0, 0, 0))
        canvas.paste(im, (0, y + 24))
        y += im.height + 24
    return np.asarray(canvas)


def _parse_json(text):
    """First JSON object in text (tolerates surrounding text / code fences)."""
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        raise ValueError("no JSON object in answer")
    return json.loads(text[s:e + 1])


def _coerce(obj, schema):
    """Fill missing keys / fix types and enums with safe defaults; returns (dict, repaired keys)."""
    out, fixed = {}, []
    for k, spec in schema["properties"].items():
        v, t = obj.get(k), spec.get("type")
        try:
            if v is None:
                raise ValueError
            if "enum" in spec:
                v = str(v).strip().lower()
                if v not in spec["enum"]:
                    raise ValueError
            elif t == "boolean":
                v = v if isinstance(v, bool) else {"true": True, "false": False}[str(v).strip().lower()]
            elif t == "integer":
                v = int(round(float(v)))
            else:
                v = str(v)
        except (ValueError, TypeError, KeyError):
            e = spec.get("enum")
            v = ("none" if e and "none" in e else "continue" if e and "continue" in e else e[0]) if e else \
                {"boolean": False, "integer": -1}.get(t, "")
            fixed.append(k)
        out[k] = v
    return out, fixed


class MoondreamAPI(object):
    """Same interface as ClaudeCLI (ask(parts, schema) -> dict; system, calls, cost_usd, latency_s).
    ask(parts, None) returns {"answer": raw text} (no JSON instructions, no repair).

    The images of a query are tiled into one labelled mosaic (the /query endpoint takes a single image);
    the text parts, the system prompt and a description of the JSON schema form the question."""

    def __init__(self, model=MOONDREAM_DEFAULT, system="", timeout=60, retries=2, reasoning=False, log=None):
        self.model, self.system, self.timeout, self.retries, self.reasoning = model, system, timeout, retries, reasoning
        self.key = _moondream_key()
        self.calls, self.cost_usd, self.latency_s, self.repairs, self.parse_failures = 0, 0.0, 0.0, 0, 0
        self.log = log

    def build_query(self, parts, schema):
        """(image, question) for one query: the images tiled into one mosaic, the text + JSON schema as the question
        (schema None: the text alone, for short free-form answers)."""
        images = [p for p in parts if not isinstance(p, str)]
        texts = [p for p in parts if isinstance(p, str)]
        labels = [f"image {i + 1}" for i in range(len(images))]
        img = images[0] if len(images) == 1 else _mosaic(images, labels)
        if len(images) > 1:
            texts.insert(0, f"The picture shows {len(images)} images stacked top to bottom, labelled "
                            f"{', '.join(labels)}.")
        if schema is None:
            return img, "\n\n".join(([self.system] if self.system else []) + texts)
        fields = "\n".join(f'  "{k}": {" | ".join(map(json.dumps, v["enum"])) if "enum" in v else v.get("type")}'
                           f'  // {v.get("description", "")}' for k, v in schema["properties"].items())
        question = "\n\n".join([self.system] + texts + [
            "Answer with ONE JSON object and nothing else, with exactly these keys:\n{\n" + fields + "\n}"])
        return img, question

    def _request(self, img, question):
        """One HTTP call -> answer text (raises urllib errors)."""
        import urllib.request

        body = {"model": self.model, "image_url": "data:image/png;base64," + png_b64(img), "question": question,
                "stream": False}
        if self.reasoning:
            body["reasoning"] = True
        req = urllib.request.Request(MOONDREAM_URL, data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json", "X-Moondream-Auth": self.key})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return str(json.loads(r.read()).get("answer", ""))

    def ask(self, parts, schema):
        import urllib.error

        img, question = self.build_query(parts, schema)
        last = None
        for attempt in range(self.retries + 1):
            t0 = time.time()
            try:
                answer = self._request(img, question)
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read()[:300]!r}"
                if e.code in (400, 401, 403):
                    break
                time.sleep(1 + attempt)
                continue
            except Exception as e:
                last = repr(e)
                time.sleep(1 + attempt)
                continue
            self.calls += 1
            self.latency_s += time.time() - t0
            if schema is None:
                return {"answer": answer.strip()}
            try:
                obj = _parse_json(answer)
            except ValueError:
                obj = {}
                self.parse_failures += 1
            res, fixed = _coerce(obj, schema)
            self.repairs += bool(fixed)
            if fixed and self.log is not None:
                self.log.info("%s answer repaired (%s): %s", self.model, ",".join(fixed), answer[:300])
            res["_raw"] = answer[:800]
            return res
        raise VLMError(f"{self.model} call failed: {last}")


# ------------------------------------------------------------------ Moondream2 (local GPU) ---------

MOONDREAM_LOCAL_PORT = 8100
MOONDREAM_LOCAL_PYTHON = pathlib.Path(__file__).resolve().parents[3] / "model" / ".venv-moondream" / "bin" / "python"


class MoondreamLocal(MoondreamAPI):
    """Moondream2 on the local GPU (model/moondream2), served by moondream_server.py over HTTP.

    Same prompt / mosaic / JSON repair as MoondreamAPI; only the transport differs. If no server answers on
    the port, one is started (its own venv: model/.venv-moondream) and stopped again at exit.
    server_latency_s: GPU-side time of the last call (image encode + prefill + decode), without HTTP/PNG."""

    def __init__(self, model="moondream2", system="", timeout=60, retries=1, max_tokens=384, port=MOONDREAM_LOCAL_PORT,
                 log=None):
        self.model, self.system, self.timeout, self.retries, self.reasoning = model, system, timeout, retries, False
        self.max_tokens, self.url = max_tokens, f"http://127.0.0.1:{port}"
        self.calls, self.cost_usd, self.latency_s, self.repairs, self.parse_failures = 0, 0.0, 0.0, 0, 0
        self.log, self.proc, self.last_server = log, None, {}
        self.info = self._ensure_server(port)

    def _health(self):
        import urllib.request

        try:
            with urllib.request.urlopen(self.url + "/healthz", timeout=2) as r:
                return json.loads(r.read())
        except Exception:
            return None

    def _ensure_server(self, port):
        import atexit
        import subprocess

        info = self._health()
        if info:
            return info
        if self.log is not None:
            self.log.info("starting moondream_server.py on port %d", port)
        self.proc = subprocess.Popen([str(MOONDREAM_LOCAL_PYTHON), str(pathlib.Path(__file__).with_name(
            "moondream_server.py")), "--port", str(port)])
        atexit.register(self.proc.terminate)
        for _ in range(300):
            if self.proc.poll() is not None:
                raise VLMError(f"moondream_server.py exited with code {self.proc.returncode}")
            info = self._health()
            if info:
                return info
            time.sleep(1)
        raise VLMError("moondream_server.py did not come up in 300 s")

    def detect(self, img, phrase):
        """Moondream2's grounded detection: boxes [{"x_min", "y_min", "x_max", "y_max"}] (0..1) of `phrase`."""
        return self._skill("/detect", "objects", img, phrase)

    def point(self, img, phrase):
        """Moondream2's pointing: points [{"x", "y"}] (0..1) of `phrase`."""
        return self._skill("/point", "points", img, phrase)

    def _skill(self, path, key, img, phrase):
        import urllib.request

        t0 = time.time()
        req = urllib.request.Request(self.url + path, data=json.dumps(
            {"image_b64": png_b64(img), "object": phrase}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read())
        if key not in out:
            raise VLMError(f"moondream_server {path}: {out.get('error', out)}")
        self.calls += 1
        self.latency_s += time.time() - t0
        self.last_server = {k: v for k, v in out.items() if k != key}
        return out[key]

    def _request(self, img, question):
        import urllib.request

        body = {"image_b64": png_b64(img), "question": question, "max_tokens": self.max_tokens, "temperature": 0.0}
        req = urllib.request.Request(self.url + "/query", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read())
        self.last_server = {k: v for k, v in out.items() if k != "answer"}
        return str(out["answer"])


def make_vlm(backend, model=None, effort=None, log=None):
    """'claude' (Claude Code CLI), 'moondream' (Moondream cloud API) or 'moondream2' (local GPU)."""
    if backend == "claude":
        return ClaudeCLI(model=model or DEFAULT_MODEL, effort=effort)
    if backend == "moondream":
        return MoondreamAPI(model=model or MOONDREAM_DEFAULT, log=log)
    if backend == "moondream2":
        return MoondreamLocal(model=model or "moondream2", log=log)
    raise ValueError(f"unknown VLM backend {backend!r}")
