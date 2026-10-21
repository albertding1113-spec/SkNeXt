import os
import yaml
import torch
from yacs.config import CfgNode as CN
from pathlib import Path
from typing import Optional, Dict
from sknext.config.config import SkNeXt_Config, update_config, load_config
from sknext.utils.utils import set_seed


class SkNeXt:
    """Configure and dispatch CUDA-backed segmentation or restoration.

    Combines YAML overrides with defaults, creates result directories, seeds
    random generators, and supplies the selected device to the workflow.
    """
    def __init__(
            self,
            config: str,
            run_id: int,
            gpu: str,
    ):
        """Load configuration and prepare a run's output directory and CUDA device.

        Args:
            config: Path to a YAML configuration file.
            run_id: Numeric identifier used in output and checkpoint paths.
            gpu: CUDA-visible device selector, such as an index or GPU UUID.
        """
        # read yaml file, write over default Config
        default_cfg = SkNeXt_Config(job_id=run_id).get_cfg_defaults()
        input_cfg = load_config(config)
        self.cfg = update_config(default_cfg, input_cfg, run_id)
        channels = list(self.cfg.TASK.CHANNELS)
        if not channels or any(not isinstance(ch, str) or
                               (ch not in {"F", "P", "C", "A", "SR"} and
                                not (ch.startswith("S.") and len(ch) > 2)) for ch in channels):
            raise ValueError("TASK.CHANNELS must contain F/P/C/A, S.<name>, or exclusively SR")
        if "SR" in channels and channels != ["SR"]:
            raise ValueError('TASK.CHANNELS: ["SR"] cannot be combined with segmentation channels')
        self.workflow = ("restoration" if channels == ["SR"] else
                         "semantic" if all(ch.startswith("S.") for ch in channels) else "segmentation")
        if bool(self.cfg.TRAIN.ENABLE) == bool(self.cfg.INFER.ENABLE):
            raise ValueError("Choose exactly one: TRAIN.ENABLE=true or INFER.ENABLE=true")
        # Semantic loss weights default to one per channel, rather than the
        # two-channel F/P defaults of the instance workflow.
        if self.workflow == "semantic" and "CHANNEL_WEIGHTS" not in input_cfg.get("TASK", {}):
            self.cfg.TASK.CHANNEL_WEIGHTS = [1.0] * len(channels)
        self.run_id = run_id
        # create result dir from config
        Path(self.cfg.PATHS.RESULT_DIR.PATH_).mkdir(parents=True, exist_ok=True)
        # setup CPU and GPU
        self.gpu = gpu
        self.setup_device()
        set_seed()


    def setup_device(self):
        """Configure the requested GPU and select the first visible CUDA device.

        Stores self.device and prints the selected GPU name. CPU mode raises
        EnvironmentError; unsupported device types or unavailable CUDA fail assertions.
        """
        self.device_type = self.cfg.SYSTEM.DEVICE.lower()
        assert self.device_type in ["cpu", "gpu", "cuda"], "unknown device type."
        if self.device_type == "cpu":
            raise EnvironmentError("CPU device not recommended.")
        elif self.device_type in ["gpu", "cuda"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        assert torch.cuda.is_available(), "CUDA device not available."
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        print(f"Using device: {self.device} \n GPU name: {torch.cuda.get_device_name(self.device)}", flush=True)


    def run(self):
        """Select the runner from TASK.CHANNELS, with mode supplied by the YAML."""
        print(f"Workflow: {self.workflow}", flush=True)
        if self.workflow == "semantic":
            from sknext.run.semantic_workflow import run
            # The semantic runner consumes ordinary nested dictionaries.
            run(yaml.safe_load(self.cfg.dump()), self.run_id, self.device)
        elif self.workflow == "restoration":
            from sknext.run.restoration_workflow import RestorationWorkflow
            self.task = RestorationWorkflow(self.cfg, self.device, self.run_id)
            self.task.run()
        else:
            from sknext.run.networkflow import Segmentation_Workflow
            self.task = Segmentation_Workflow(self.cfg, self.device, self.run_id)
            self.task.run()
