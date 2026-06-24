import os
import yaml
import torch
from yacs.config import CfgNode as CN
from pathlib import Path
from typing import Optional, Dict
from sknext.config.config import SkNeXt_Config, update_config, load_config
from sknext.utils.utils import set_seed
from sknext.run.workflow import Segmentation_Workflow


class SkNeXt:
    def __init__(
            self,
            config: str,
            run_id: int,
            gpu: str,
    ):
        # read yaml file, write over default Config
        default_cfg = SkNeXt_Config(job_id=run_id).get_cfg_defaults()
        input_cfg = load_config(config)
        self.cfg = update_config(default_cfg, input_cfg, run_id)
        self.run_id = run_id
        # create result dir from config
        Path(self.cfg.PATHS.RESULT_DIR.PATH_).mkdir(parents=True, exist_ok=True)
        # set seed before model
        set_seed()
        # setup CPU and GPU
        self.gpu = gpu
        self.setup_device()


    def setup_device(self):
        self.device_type = self.cfg.SYSTEM.DEVICE.lower()
        assert self.device_type in ["cpu", "gpu", "cuda"], "unknown device type."
        if self.device_type == "cpu":
            raise EnvironmentError("CPU device not recommended.")
        elif self.device_type in ["gpu", "cuda"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        assert torch.cuda.is_available(), "CUDA device not available."
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        print(f"Using device: {self.device} \n GPU name: {torch.cuda.get_device_name(self.device)}")


    def run(self):
        # determine task type 1. SEMANTIC 2. INSTANCE
        self.task = Segmentation_Workflow(self.cfg, self.device, self.run_id)
        self.task.run()
