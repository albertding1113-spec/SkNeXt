"""Tk desktop launcher. The GUI never imports the training runtime."""
from __future__ import annotations

import ast
import csv
import io
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from yacs.config import CfgNode
from sknext.config.config import SkNeXt_Config, load_config, update_config


DERIVED = {
    "SYSTEM.DEVICE", "TRAIN.ENABLE", "INFER.ENABLE", "PATHS.CHECKPOINT_DIR",
    "DATA.INFER.INFER_LOG",
}
CHOICES = {
    "TRAIN.OPTIMIZER": ("ADAMW", "ADAM", "SGD"),
    "TRAIN.LR_SCHEDULER.NAME": ("warmupcosine", "one_cycle"),
    "DATA.NORMALIZATION.TYPE": ("zero_mean_unit_variance", "scale_range"),
    "MODEL.ARCHITECTURE": ("unext_v2", "u_next_v2", "unext-v2"),
}


def leaves(node, prefix=""):
    """Yield editable configuration leaves as (dotted_path, value) pairs.

    Args:
        node: Configuration node to traverse recursively.
        prefix: Parent key path, empty for the root.

    Derived paths and workflow/device flags are excluded from the form.
    """
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, CfgNode):
            yield from leaves(value, path)
        elif path not in DERIVED and not (
            path.startswith("PATHS.RESULT_DIR.") and path != "PATHS.RESULT_DIR.PATH"
        ):
            yield path, value


def set_value(cfg, path, value):
    """Assign value to an existing dotted path in cfg, modifying cfg in place."""
    parts = path.split(".")
    for part in parts[:-1]:
        cfg = cfg[part]
    cfg[parts[-1]] = value


def parse_value(text, default):
    """Convert a form value to the type of its configuration default.

    Args:
        text: Entry text, or a boolean supplied by a Tk BooleanVar.
        default: Reference value defining the expected scalar or sequence type.

    Returns:
        Parsed value; strings remain literal and sequences preserve the default type.

    Raises:
        ValueError: The text is not a supported literal or has an incompatible type.
    """
    if isinstance(default, str):
        return text
    if isinstance(default, bool):
        return bool(text)
    try:
        value = ast.literal_eval(text)
        if isinstance(default, (tuple, list)):
            if not isinstance(value, (tuple, list)):
                raise ValueError("expected a list such as [1, 2, 3]")
            return type(default)(value)
        if isinstance(default, float) and type(value) in (int, float):
            if not math.isfinite(value):
                raise ValueError("expected a finite number")
            return float(value)
        if type(value) is not type(default):
            raise ValueError(f"expected {type(default).__name__}")
        return value
    except (SyntaxError, TypeError) as error:
        raise ValueError("use Python values, e.g. [20, 256, 256, 1]") from error


def validate(cfg, run_id, workdir):
    """Reject common configuration errors before creating a worker."""
    def require(condition, message):
        """Raise ValueError with message when the validation condition is false."""
        if not condition:
            raise ValueError(message)

    def vector(value, length, predicate, name):
        """Check value has length entries satisfying predicate; name labels errors."""
        require(len(value) == length and all(predicate(x) for x in value),
                f"{name}: expected {length} valid values.")

    positive_int = lambda x: type(x) is int and x > 0
    vector(cfg.DATA.PATCH_SIZE, 4, positive_int, "Patch size (Z, Y, X, C)")
    for split in (cfg.DATA.TRAIN, cfg.DATA.VAL, cfg.DATA.INFER):
        vector(split.OVERLAP, 3, lambda x: type(x) in (int, float) and 0 <= x < 1, "Overlap")
        vector(split.PADDING, 3, lambda x: type(x) is int and x >= 0, "Padding")
    require(positive_int(cfg.TRAIN.BATCH_SIZE), "Batch size must be a positive integer.")
    require(positive_int(cfg.TRAIN.EPOCHS), "Epochs must be a positive integer.")
    require(cfg.TRAIN.LR > 0, "Learning rate must be positive.")
    require(cfg.TRAIN.W_DECAY >= 0, "Weight decay cannot be negative.")
    require(bool(cfg.TASK.CHANNELS), "Choose at least one task channel.")
    levels = len(cfg.MODEL.FEATURE_MAPS)
    require(levels >= 2, "Model requires at least two feature levels.")
    for key in ("ISOTROPY", "CONVNEXT_LAYERS"):
        require(len(cfg.MODEL[key]) == levels, f"MODEL.{key} must match FEATURE_MAPS length.")
    for key in ("Z_DOWN", "YX_DOWN"):
        vector(cfg.MODEL[key], levels - 1, positive_int, f"MODEL.{key}")
    vector(cfg.DATA.INFER.BLOCK_FACTOR, 3, positive_int, "Block factor")
    vector(cfg.DATA.INFER.BLOCK_CENTRAL_FACTOR, 3, positive_int, "Central block factor")
    require(all(a <= b for a, b in zip(cfg.DATA.INFER.BLOCK_CENTRAL_FACTOR,
                                      cfg.DATA.INFER.BLOCK_FACTOR)),
            "Central block factors cannot exceed block factors.")
    clip = cfg.DATA.NORMALIZATION.PERC_CLIP
    if clip.ENABLE:
        require(0 <= clip.LOWER_PERC < clip.UPPER_PERC <= 100,
                "Percentile clipping requires 0 <= lower < upper <= 100.")
    paths = [cfg.DATA.TRAIN.PATH, cfg.DATA.TRAIN.GT_PATH,
             cfg.DATA.VAL.PATH, cfg.DATA.VAL.GT_PATH] if cfg.TRAIN.ENABLE else [cfg.DATA.INFER.PATH]
    if cfg.INFER.ENABLE:
        # The current inference loop always uses SkeletonManager and SWC seeds.
        require(cfg.INFER.SKELETON.ENABLE, "Current inference requires skeleton seeds.")
        paths.append(cfg.DATA.INFER.SKELETON_PATH)
        require(len(cfg.DATA.INFER.CHANNEL) == cfg.DATA.PATCH_SIZE[3],
                "Inference channel count must match patch size C.")
    for path in paths:
        require(bool(path.strip()) and (workdir / path).exists(), f"Input path does not exist: {path}")
    if cfg.INFER.ENABLE or cfg.MODEL.LOAD_CHECKPOINT:
        checkpoint = workdir / cfg.PATHS.CHECKPOINT_DIR / f"checkpoint_{run_id:02d}.pth"
        require(checkpoint.is_file(), f"Checkpoint does not exist: {checkpoint}")


def discover_gpus():
    """Query NVIDIA GPU identifiers and free/total memory through nvidia-smi.

    Returns:
        Mapping of human-readable GPU descriptions to stable device UUIDs.

    Raises:
        OSError: The NVIDIA utility cannot be launched.
        subprocess.SubprocessError: The query fails or exceeds its timeout.
    """
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.free,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=True,
    )
    devices = {}
    for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
        if len(row) == 5:
            index, uuid, name, free, total = (item.strip() for item in row)
            devices[f"GPU {index} · {name} · {free}/{total} MiB free"] = uuid
    return devices


class Application(ttk.Frame):
    """Desktop configuration editor and subprocess launcher for SkNeXt.

    Owns Tk variables, GPU discovery, YAML persistence, and workflow controls.
    Background threads send events through a queue; Tk widgets are updated on
    its main thread. Each worker gets a configuration snapshot and disk log.
    """
    def __init__(self, root, config=None):
        """Build the editor under root and optionally load the YAML file config.

        Initializes default settings, starts GPU discovery, and schedules event
        polling. The training runtime is loaded only by a launched worker.
        """
        super().__init__(root, padding=16)
        self.pack(fill="both", expand=True)
        self.root = root
        self.cfg = SkNeXt_Config(0).get_cfg_defaults()
        self.defaults = dict(leaves(self.cfg))
        self.variables = {}
        self.events = queue.Queue(maxsize=2000)
        self.process = None
        self.stopping = False
        self.closing = False
        self.devices = {}
        self.workflow = tk.StringVar(value="Training")
        self.run_id = tk.StringVar(value="0")
        self.workdir = tk.StringVar(value=str(Path.cwd()))
        self.gpu = tk.StringVar()
        self.status = tk.StringVar(value="Ready — load a YAML config or edit the defaults.")
        self._build()
        self.populate(self.cfg)
        if config:
            self.load(config)
        self.refresh_gpus()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)

    def _build(self):
        """Create configuration tabs, path selectors, workflow controls, and the log view."""
        ttk.Label(self, text="SkNeXt", font=("Segoe UI", 23, "bold")).pack(anchor="w")
        ttk.Label(self, text="Training & inference workspace").pack(anchor="w", pady=(0, 12))
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x")
        for label, action in (("Load YAML…", self.load), ("Save YAML…", self.save)):
            ttk.Button(toolbar, text=label, command=action).pack(side="left", padx=(0, 8))
        ttk.Label(toolbar, text="Workflow").pack(side="left", padx=(16, 6))
        ttk.Combobox(toolbar, textvariable=self.workflow, values=("Training", "Inference"),
                     state="readonly", width=12).pack(side="left")
        ttk.Label(toolbar, text="Run ID").pack(side="left", padx=(16, 6))
        ttk.Entry(toolbar, textvariable=self.run_id, width=8).pack(side="left")
        row = ttk.Frame(self)
        row.pack(fill="x", pady=10)
        ttk.Label(row, text="Working directory").pack(side="left", padx=(0, 8))
        ttk.Entry(row, textvariable=self.workdir).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse…", command=lambda: self.browse(self.workdir)).pack(side="left")
        row = ttk.Frame(self)
        row.pack(fill="x")
        ttk.Label(row, text="CUDA GPU").pack(side="left", padx=(0, 8))
        self.gpu_box = ttk.Combobox(row, textvariable=self.gpu, state="readonly")
        self.gpu_box.pack(side="left", fill="x", expand=True)
        self.refresh_button = ttk.Button(row, text="Refresh GPUs", command=self.refresh_gpus)
        self.refresh_button.pack(side="left")
        ttk.Label(self, text="Paths resolve against the working directory. Lists use [a, b, c]; booleans use True/False."
                  ).pack(anchor="w", pady=(10, 4))
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)
        groups = {}
        for path, default in self.defaults.items():
            group = path.split(".")[0]
            if group not in groups:
                tab = ttk.Frame(notebook)
                notebook.add(tab, text=group.title())
                canvas = tk.Canvas(tab, highlightthickness=0)
                scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
                scrollbar.pack(side="right", fill="y")
                canvas.pack(side="left", fill="both", expand=True)
                canvas.configure(yscrollcommand=scrollbar.set)
                content = ttk.Frame(canvas, padding=12)
                window = canvas.create_window((0, 0), window=content, anchor="nw")
                content.bind("<Configure>", lambda e, c=canvas: c.configure(scrollregion=c.bbox("all")))
                canvas.bind("<Configure>", lambda e, c=canvas, w=window: c.itemconfigure(w, width=e.width))
                content.columnconfigure(1, weight=1)
                groups[group] = (content, 0)
            content, index = groups[group]
            groups[group] = content, index + 1
            label = path.split(".", 1)[1].replace("_", " ").title()
            ttk.Label(content, text=label, width=38).grid(row=index, column=0, sticky="w", pady=5)
            variable = tk.BooleanVar() if isinstance(default, bool) else tk.StringVar()
            self.variables[path] = variable
            if isinstance(default, bool):
                widget = ttk.Checkbutton(content, variable=variable)
            elif path in CHOICES:
                widget = ttk.Combobox(content, textvariable=variable, values=CHOICES[path], state="readonly")
            else:
                widget = ttk.Entry(content, textvariable=variable)
            widget.grid(row=index, column=1, sticky="ew", padx=8)
            if path.endswith("PATH"):
                ttk.Button(content, text="Folder…", command=lambda v=variable: self.browse(v)).grid(row=index, column=2)
                if path == "DATA.INFER.PATH":
                    ttk.Button(content, text="File…", command=lambda v=variable: self.browse(v, True)).grid(row=index, column=3)
        ttk.Label(self, text="Checkpoints: <result path>/results/<run ID>/checkpoints/checkpoint_<ID with 2 digits>.pth"
                  ).pack(anchor="w", pady=(8, 3))
        controls = ttk.Frame(self)
        controls.pack(fill="x", pady=6)
        self.start_button = ttk.Button(controls, text="Start workflow", command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        ttk.Label(controls, textvariable=self.status).pack(side="left", padx=8)
        self.log = ScrolledText(self, height=10, state="disabled", font=("Consolas", 9))
        self.log.pack(fill="x")

    def browse(self, variable, file=False):
        """Open a file or folder picker and store its selection in variable.

        Args:
            variable: Tk variable receiving the selected absolute path.
            file: Open a file picker when true, otherwise a directory picker.

        Cancelling leaves the variable unchanged.
        """
        path = filedialog.askopenfilename(parent=self.root) if file else filedialog.askdirectory(parent=self.root)
        if path:
            variable.set(path)

    def populate(self, cfg):
        """Replace the form's configuration values and workflow selection with cfg."""
        self.cfg = cfg
        for path, value in leaves(cfg):
            self.variables[path].set(value if isinstance(value, (str, bool)) else repr(value))
        self.workflow.set("Inference" if cfg.INFER.ENABLE and not cfg.TRAIN.ENABLE else "Training")

    def collect(self):
        """Parse the form and return (configuration, run_id) for saving or launching.

        Recomputes derived paths, selects exactly one workflow, and sets GPU mode.
        Raises ValueError for invalid field values or a negative/noninteger run ID.
        """
        run_id = int(self.run_id.get())
        if run_id < 0:
            raise ValueError("Run ID cannot be negative.")
        cfg = self.cfg.clone()
        for path, variable in self.variables.items():
            try:
                set_value(cfg, path, parse_value(variable.get(), self.defaults[path]))
            except ValueError as error:
                raise ValueError(f"{path}: {error}") from error
        cfg.TRAIN.ENABLE = self.workflow.get() == "Training"
        cfg.INFER.ENABLE = not cfg.TRAIN.ENABLE
        cfg.SYSTEM.DEVICE = "gpu"
        return update_config(cfg, CfgNode(), run_id), run_id

    def load(self, path=None):
        """Load YAML from path, or prompt for a file when path is omitted.

        Merges overrides into fresh defaults and validates types before replacing
        the form. Load failures are displayed in a dialog.
        """
        path = path or filedialog.askopenfilename(filetypes=[("YAML config", "*.yaml *.yml")])
        if not path:
            return
        try:
            cfg = update_config(SkNeXt_Config(0).get_cfg_defaults(), load_config(path), 0)
            # Parse all fields before replacing the current form.
            for key, value in leaves(cfg):
                if isinstance(self.defaults[key], bool) and type(value) is not bool:
                    raise ValueError(f"{key}: expected a boolean")
                parse_value(value if isinstance(value, (str, bool)) else repr(value), self.defaults[key])
            self.populate(cfg)
            self.status.set(f"Loaded {Path(path).name}")
        except Exception as error:
            messagebox.showerror("Cannot load configuration", str(error), parent=self.root)

    def save(self):
        """Prompt for a YAML destination and write the current parsed configuration.

        Updates the status on success and displays parsing or I/O errors in a dialog.
        """
        try:
            cfg, _ = self.collect()
            path = filedialog.asksaveasfilename(defaultextension=".yaml", filetypes=[("YAML config", "*.yaml")])
            if path:
                Path(path).write_text(cfg.dump(), encoding="utf-8")
                self.status.set(f"Saved {Path(path).name}")
        except Exception as error:
            messagebox.showerror("Cannot save configuration", str(error), parent=self.root)

    def refresh_gpus(self):
        """Start GPU discovery in a background thread and disable refresh until it finishes."""
        self.refresh_button.configure(state="disabled")
        self.status.set("Detecting NVIDIA GPUs…")
        def detect():
            """Queue detected devices or a GPU-query error without touching Tk widgets."""
            try:
                self.events.put(("gpus", discover_gpus()))
            except Exception as error:
                self.events.put(("gpu_error", str(error)))
        threading.Thread(target=detect, daemon=True).start()

    def start(self):
        """Validate the form and launch one workflow using the current Python interpreter.

        Saves a YAML snapshot and console log in the run directory, sets the GPU
        UUID in the child environment, and starts asynchronous output collection.
        Startup errors are displayed in a dialog; an active worker blocks another launch.
        """
        if self.process is not None:
            return
        try:
            cfg, run_id = self.collect()
            workdir = Path(self.workdir.get()).expanduser().resolve()
            if not workdir.is_dir():
                raise ValueError("Choose an existing working directory.")
            if self.gpu.get() not in self.devices:
                raise ValueError("Select an available NVIDIA GPU. Refresh after installing a working NVIDIA driver.")
            validate(cfg, run_id, workdir)
            output = workdir / cfg.PATHS.RESULT_DIR.PATH_
            output.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            snapshot = output / f"gui-{stamp}.yaml"
            snapshot.write_text(cfg.dump(), encoding="utf-8")
            log_path = output / f"gui-{stamp}.log"
            env = os.environ.copy()
            device = self.devices[self.gpu.get()]
            env["CUDA_VISIBLE_DEVICES"] = device
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
            command = [sys.executable, "-u", "-m", "sknext", "--config", str(snapshot), "--run_id", str(run_id), "--gpu", device]
            log_file = log_path.open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(command, cwd=workdir, env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except Exception:
                log_file.close()
                raise
            self.process = process
            self.stopping = False
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.status.set(f"Running {self.workflow.get().lower()} · PID {process.pid}")
            self.append_log(f"\nConfiguration: {snapshot}\nLog: {log_path}\n")
            def read_output():
                """Copy worker output to disk and the GUI queue, then queue its exit code.

                Closes the output stream and log file when reading completes.
                """
                try:
                    with log_file, process.stdout:
                        for line in process.stdout:
                            log_file.write(line)
                            log_file.flush()
                            self.events.put(("log", line))
                finally:
                    self.events.put(("exit", process.wait()))
            threading.Thread(target=read_output, daemon=True).start()
        except Exception as error:
            messagebox.showerror("Cannot start workflow", str(error), parent=self.root)

    def append_log(self, text):
        """Append text to the console, trim old display lines, and scroll to the end."""
        self.log.configure(state="normal")
        self.log.insert("end", text)
        if int(self.log.index("end-1c").split(".")[0]) > 4000:
            self.log.delete("1.0", "1000.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def poll(self):
        """Process queued GPU, console, and completion events on the Tk thread.

        Limits work per callback to keep the interface responsive, restores launch
        controls after exit, and completes a pending window-close request.
        """
        for _ in range(200):
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind in ("gpus", "gpu_error"):
                self.refresh_button.configure(state="normal")
                self.devices = payload if kind == "gpus" else {}
                self.gpu_box.configure(values=list(self.devices))
                if self.gpu.get() not in self.devices:
                    self.gpu.set(next(iter(self.devices), ""))
                if not self.process:
                    self.status.set("Ready" if self.devices else "No NVIDIA GPUs detected")
                if kind == "gpu_error":
                    self.append_log(f"GPU detection: {payload}\n")
            elif kind == "log":
                self.append_log(payload)
            elif kind == "exit":
                self.process = None
                self.start_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                status = "Stopped" if self.stopping else ("Completed" if payload == 0 else f"Failed (exit {payload})")
                self.status.set(status)
                self.append_log(f"\n{status}\n")
                if self.closing:
                    self.root.destroy()
                    return
        self.root.after(100, self.poll)

    def stop(self):
        """Terminate the active worker and schedule a forced kill after five seconds.

        Updates the interface to show cancellation; no extra checkpoint is saved.
        """
        process = self.process
        if process is None or process.poll() is not None:
            return
        self.stopping = True
        self.status.set("Stopping…")
        self.stop_button.configure(state="disabled")
        process.terminate()
        def kill_if_needed():
            """Kill the captured worker if it is still running after the termination timeout."""
            if process.poll() is None:
                process.kill()
        self.root.after(5000, kill_if_needed)

    def close(self):
        """Close the window, asking to stop an active workflow before destroying Tk."""
        if self.process is not None:
            if not messagebox.askyesno("Stop running workflow?", "Closing will stop the workflow. The current batch may be incomplete.", parent=self.root):
                return
            self.closing = True
            self.stop()
        else:
            self.root.destroy()


def main(config=None):
    """Create the desktop window and run its Tk event loop.

    Args:
        config: Optional YAML path to preload into the editor.
    """
    root = tk.Tk()
    root.title("SkNeXt — Training & Inference")
    root.geometry("1120x850")
    root.minsize(850, 650)
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    Application(root, config)
    root.mainloop()


if __name__ == "__main__":
    main()
