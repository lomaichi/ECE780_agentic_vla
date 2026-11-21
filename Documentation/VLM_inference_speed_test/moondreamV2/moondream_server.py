"""Local Moondream2 inference server for the VLM speed test (GPU, HTTP on localhost).

The LIBERO client venv is py3.8 / torch 1.11 (no RTX 5090 support), so Moondream2 runs in its own venv
(py3.11, torch cu128) and the judge talks to it over HTTP, like pi0 over its websocket server.
vlm_backends.MoondreamLocal starts this server by itself if nothing answers on the port.

  /home/DockerShared/ECE780/model/.venv-moondream/bin/python moondream_server.py [--port 8100] [--compile]

  GET  /healthz  -> {"model", "revision", "device", "max_context"}
  POST /query    {"image_b64": png base64, "question": str, "max_tokens": int, "temperature": float,
                  "reasoning": bool}
              -> {"answer", "latency_s", "encode_s", "prompt_tokens", "answer_tokens"}
  POST /detect   {"image_b64": png base64, "object": str}
              -> {"objects": [{"x_min", "y_min", "x_max", "y_max"} (0..1)], "latency_s", "encode_s"}
  POST /point    {"image_b64": png base64, "object": str}
              -> {"points": [{"x", "y"} (0..1)], "latency_s", "encode_s"}

Weights: ECE780/model/moondream2 (vikhyatk/moondream2, revision 2025-06-21), tokenizer in ECE780/model/hf_cache;
loaded offline.
"""

import argparse
import base64
import io
import json
import os
import pathlib
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL_ROOT = pathlib.Path(__file__).resolve().parents[3] / "model"
# the model code loads its tokenizer (moondream/starmie-v1) from the HF hub: use the copy cached in model/
os.environ.setdefault("HF_HOME", str(MODEL_ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

MODEL_DIR = MODEL_ROOT / "moondream2"
MODEL_NAME = "moondream2"


def load(model_dir, compile_model):
    # import the model code straight from the weights folder (as a namespace package) instead of
    # trust_remote_code, whose module cache misses moondream2's nested relative imports
    import importlib
    import sys

    sys.path.insert(0, str(model_dir.parent))
    hf = importlib.import_module(f"{model_dir.name}.hf_moondream")
    model = hf.HfMoondream.from_pretrained(str(model_dir), local_files_only=True, torch_dtype=torch.bfloat16,
                                           device_map={"": "cuda"})
    model.eval()
    if compile_model:
        model.model.compile()
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--model-dir", type=pathlib.Path, default=MODEL_DIR)
    p.add_argument("--compile", action="store_true", help="torch.compile the vision encoder / decoder")
    a = p.parse_args()

    t0 = time.time()
    model = load(a.model_dir, a.compile)
    md = model.model
    tok = md.tokenizer
    revision = (a.model_dir / "SOURCE.txt").read_text().strip() if (a.model_dir / "SOURCE.txt").exists() else ""
    # warm-up (CUDA kernels, KV caches, compile)
    with torch.inference_mode():
        for _ in range(2):
            # encode separately: encode_image(PIL, settings) needs a "variant" key in settings (model bug)
            enc = model.encode_image(Image.new("RGB", (512, 512), (128, 128, 128)))
            model.query(enc, "What is in the image?", settings={"max_tokens": 8, "temperature": 0.0})
    print(f"{MODEL_NAME} ready on cuda in {time.time() - t0:.1f} s, port {a.port}", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            if self.path == "/healthz":
                self._send(200, {"model": MODEL_NAME, "revision": revision, "device": torch.cuda.get_device_name(),
                                 "max_context": md.config.text.max_context})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in ("/query", "/detect", "/point"):
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                img = Image.open(io.BytesIO(base64.b64decode(req["image_b64"]))).convert("RGB")
                if self.path in ("/detect", "/point"):
                    skill, key = (model.detect, "objects") if self.path == "/detect" else (model.point, "points")
                    with torch.inference_mode():
                        torch.cuda.synchronize()
                        t0 = time.time()
                        enc = model.encode_image(img)
                        torch.cuda.synchronize()
                        t_enc = time.time() - t0
                        objs = skill(enc, req["object"])[key]
                        torch.cuda.synchronize()
                    return self._send(200, {key: [{k: float(v) for k, v in o.items()} for o in objs],
                                            "latency_s": time.time() - t0, "encode_s": t_enc})
                question = req["question"]
                settings = {"max_tokens": int(req.get("max_tokens", 384)),
                            "temperature": float(req.get("temperature", 0.0))}
                n_prompt = len(tok.encode(question).ids)
                with torch.inference_mode():
                    torch.cuda.synchronize()
                    t0 = time.time()
                    enc = model.encode_image(img)
                    torch.cuda.synchronize()
                    t_enc = time.time() - t0
                    budget = md.config.text.max_context - enc.pos - n_prompt - 8
                    if budget < 16:
                        raise ValueError(f"prompt too long: image {enc.pos} + question {n_prompt} tokens "
                                         f"(max_context {md.config.text.max_context})")
                    settings["max_tokens"] = min(settings["max_tokens"], budget)
                    out = model.query(enc, question, reasoning=bool(req.get("reasoning", False)), settings=settings)
                    torch.cuda.synchronize()
                    dt = time.time() - t0
                answer = out["answer"]
                self._send(200, {"answer": answer, "latency_s": dt, "encode_s": t_enc, "prompt_tokens": n_prompt,
                                 "image_tokens": enc.pos, "answer_tokens": len(tok.encode(answer).ids)})
            except Exception as e:
                self._send(400, {"error": repr(e)[:500]})

    HTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
