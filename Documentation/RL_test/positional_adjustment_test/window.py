"""Tk window for interactive_pi0.py (Python 3.11 stdlib only; start it with ./run.sh).

Launches the simulation (interactive_pi0.py, LIBERO venv) as a child process, shows the frames it sends
and forwards the keyboard to it. All command-line arguments are passed through to the simulation.
Keys and protocol: see interactive_pi0.py.
"""

import json
import os
import pathlib
import socket
import struct
import subprocess
import sys
import threading
import tkinter as tk

HERE = pathlib.Path(__file__).resolve().parent
SIM_PYTHON = "/opt/openpi/examples/libero/.venv/bin/python"
COLORS = {"PI0": "#2563eb", "HUMAN": "#dc2626", "SUCCESS": "#16a34a"}
ARROWS = {"Up": "up", "Down": "down", "Left": "left", "Right": "right"}
MODS = {"Control_L": "ctrl", "Control_R": "ctrl", "Shift_L": "shift", "Shift_R": "shift"}
COMMANDS = {"r": "reset", "n": "next", "q": "quit", "Escape": "quit"}
HELP = "space interrupt / hand back    arrows move    ctrl+up/down height    ctrl+left/right rotate    shift fine    r restart    n next init    q quit"


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return bytes(buf)


def main():
    parent, child = socket.socketpair()
    env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"))
    proc = subprocess.Popen([SIM_PYTHON, str(HERE / "interactive_pi0.py"), "--ipc-fd", str(child.fileno()), *sys.argv[1:]],
                            pass_fds=[child.fileno()], env=env)
    child.close()

    latest = {"frame": None, "seq": 0, "eof": False}
    lock = threading.Lock()

    def reader():
        try:
            while True:
                hl, pl = struct.unpack("!II", recv_exact(parent, 8))
                head = json.loads(recv_exact(parent, hl))
                payload = recv_exact(parent, pl)
                with lock:
                    latest["frame"], latest["seq"] = (head, payload), latest["seq"] + 1
        except (EOFError, OSError):
            with lock:
                latest["eof"] = True

    threading.Thread(target=reader, daemon=True).start()

    def send(msg):
        try:
            parent.sendall((json.dumps(msg) + "\n").encode())
        except OSError:
            pass

    root = tk.Tk()
    root.title("LIBERO - pi0 + keyboard (starting...)")
    root.configure(bg="#111")
    border = tk.Frame(root, bg=COLORS["PI0"], padx=5, pady=5)
    border.grid(row=0, column=0, padx=8, pady=8)
    wrist_lbl = tk.Label(border, bg="#000", bd=0, width=80, height=30, fg="#aaa",
                         text="loading LIBERO and connecting to pi0...")
    wrist_lbl.pack()
    agent_lbl = tk.Label(root, bg="#000", bd=0)
    status = tk.Label(root, bg="#111", fg="#ddd", font=("DejaVu Sans Mono", 12), anchor="w", justify="left", text=HELP)
    status.grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 8))

    images, pending, held, shown = {}, {}, set(), [0]

    def photo(data, h, w):
        return tk.PhotoImage(data=b"P6 %d %d 255 " % (w, h) + data, format="PPM")

    def refresh():
        with lock:
            seq, frame, eof = latest["seq"], latest["frame"], latest["eof"]
        if eof:
            root.destroy()
            return
        if frame is not None and seq != shown[0]:
            shown[0] = seq
            head, payload = frame
            off = 0
            for i, (h, w) in enumerate(head["sizes"]):
                key, lbl = ("w", wrist_lbl) if i == 0 else ("a", agent_lbl)
                images[key] = photo(payload[off:off + h * w * 3], h, w)
                lbl.configure(image=images[key], width=w, height=h, text="")
                off += h * w * 3
            if len(head["sizes"]) > 1 and not agent_lbl.winfo_ismapped():
                agent_lbl.grid(row=0, column=1, padx=(0, 8), pady=8, sticky="n")
            border.configure(bg=COLORS[head["mode"]])
            root.title(f"LIBERO - {head['prompt']}")
            status.configure(text=f"{head['mode']:<8} step {head['step']:<5} init {head['init']:<3} "
                                  f"gripper {head['gripper']:<7} interrupts {head['takeovers']}\n"
                                  f"{head['prompt']}\n{HELP}")
        root.after(10, refresh)

    def name(e):
        return ARROWS.get(e.keysym) or MODS.get(e.keysym) or ("space" if e.keysym == "space" else None)

    def press(e):
        k = name(e)
        if k is None:
            if e.keysym in COMMANDS:
                send({"cmd": COMMANDS[e.keysym]})
            return
        if k in pending:  # X11 auto-repeat sends release+press pairs while a key is held: ignore the repeat
            root.after_cancel(pending.pop(k))
            return
        if k in held:
            return
        held.add(k)
        if k == "space":
            send({"toggle": 1})
            return
        if e.state & 0x4:  # ctrl went down while the window lacked focus
            held.add("ctrl")
        send({"keys": sorted(held - {"space"})})

    def release(e):
        k = name(e)
        if k is None:
            return

        def really():
            pending.pop(k, None)
            held.discard(k)
            send({"keys": sorted(held - {"space"})})

        pending[k] = root.after(30, really)

    def focus_out(_):
        for aid in pending.values():
            root.after_cancel(aid)
        pending.clear()
        held.clear()
        send({"keys": []})

    def close():
        send({"cmd": "quit"})
        root.destroy()

    root.bind("<KeyPress>", press)
    root.bind("<KeyRelease>", release)
    root.bind("<FocusOut>", focus_out)
    root.protocol("WM_DELETE_WINDOW", close)
    root.after(10, refresh)
    root.focus_force()
    try:
        root.mainloop()
    finally:
        parent.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    if any(x in sys.argv[1:] for x in ("-h", "--help", "--selftest")):  # no window needed
        os.execv(SIM_PYTHON, [SIM_PYTHON, str(HERE / "interactive_pi0.py"), *sys.argv[1:]])
    main()
