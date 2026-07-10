from yacs.config import CfgNode as CN
import os
import copy
from typing import Dict, Optional
from pathlib import Path

"""
待做的事情
2. 正式在集群上跑起来SkNeXt的训练
3. 写出ims / hdf5 / ome-zarr文件的reader。ome-zarr格式兼容主流软件
4. 写出ome-zarr文件的writer
5. 写出完整的infer流程
"""

class SkNeXt_Config:
    def __init__(self, job_id:Optional[int|str]):
        # 1. SYSTEM
        _C = CN()
        _C.SYSTEM = CN()
        _C.SYSTEM.DEVICE = 'gpu'
        # 2. PATH
        _C.PATHS = CN()
        _C.PATHS.RESULT_DIR = CN()
        _C.PATHS.RESULT_DIR.PATH = './result'
        _C.PATHS.RESULT_DIR.PATH_ = os.path.join(_C.PATHS.RESULT_DIR.PATH, "results", str(job_id))
        _C.PATHS.RESULT_DIR.OUTPUT_LOG = os.path.join(_C.PATHS.RESULT_DIR.PATH_, "log")
        _C.PATHS.RESULT_DIR.OUTPUT_CHART = os.path.join(_C.PATHS.RESULT_DIR.PATH_, "chart")
        _C.PATHS.RESULT_DIR.OUTPUT_RAW = os.path.join(_C.PATHS.RESULT_DIR.PATH_, "output_raw")
        _C.PATHS.RESULT_DIR.OUTPUT_INSTANCES = os.path.join(_C.PATHS.RESULT_DIR.PATH_, "output_instances")
        _C.PATHS.RESULT_DIR.OUTPUT_POST_PROCESSING = os.path.join(_C.PATHS.RESULT_DIR.PATH_, "output_post_processing")
        # Name of the folder where weights files will be stored/loaded from.
        _C.PATHS.CHECKPOINT_DIR = os.path.join(_C.PATHS.RESULT_DIR.PATH, "checkpoints")

        # 3. TASK
        _C.TASK = CN()
        # Options: "F", "P", "C", "A", "S.[channel]".
        # "F" stands for "Foreground".
        # "P" stands for "Center part".
        # "C" stands for "Contour".
        # "A" stands for "Affinity".
        # "S.[channel]" stands for semantic segmentation channel. Example: S.spine, S.varicosity, S.soma, S.dendrite, S.axon
        _C.TASK.CHANNELS = ['F', 'P']
        # "F" channel options:
        #   "dilation": int. Default: 0.
        #   "erosion": int. Default: 0.
        # "P" channel options:
        #   "type": str. "skeleton", "centroid". Default: "skeleton".
        #   "dilation": int. Default: 0.
        #   "erosion": int. Default: 0.
        # "C" channel options:
        #   "mode": str. "thick", "inner", "outer". Default: "inner".
        #   https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.find_boundaries
        # "A" channel options"
        #   "z_affinities": list of int, Default: [1]
        #   "y_affinities": list of int, Default: [1]
        #   "x_affinities": list of int, Default: [1]
        #   The length of x/y/z affinity lists should be equal.
        # "F" channel options:
        #   "dilation": int. Default: 0.
        #   "erosion": int. Default: 0.
        _C.TASK.CHANNELS_EXTRA_OPTS = [{}]
        # Weights to calculate loss when training network.
        _C.TASK.CHANNEL_WEIGHTS = (1, 1)
        # 2.2.1. WATERSHED
        _C.TASK.WATERSHED = CN()
        # List of channels to create seed. if not provided will be determined automatically.
        _C.TASK.WATERSHED.SEED_CHANNELS = []
        # Thresholds for the seed channels. List of float between 0 and 1.
        # If not provided or "auto" will be automatically calculated based on Otsu thresholding.
        _C.TASK.WATERSHED.SEED_CHANNELS_THRESH = []
        # Channel as the topographic surface to grow the seeds. If not provided will be determined automatically.
        _C.TASK.WATERSHED.TOPOGRAPHIC_SURFACE_CHANNEL = ""
        # List of channels to be set as growth masks.
        _C.TASK.WATERSHED.GROWTH_MASK_CHANNELS = []
        # Thresholds for growth masks. List of float between 0 and 1.
        # If not provided or "auto" will be automatically calculated.
        _C.TASK.WATERSHED.GROWTH_MASK_CHANNELS_THRESH = []

        # 4. DATA
        _C.DATA = CN()
        # Patch size, "ZYXC"
        _C.DATA.PATCH_SIZE = (20, 256, 256, 1)
        # Whether to reshape the dimensions that does not satisfy the patch shape selected by padding it with reflect.
        _C.DATA.REFLECT_TO_COMPLETE_SHAPE = True
        # 3.1. NORMALIZATION
        _C.DATA.NORMALIZATION = CN()
        # Whether to apply a percentage clip before normalization.
        _C.DATA.NORMALIZATION.PERC_CLIP = CN()
        _C.DATA.NORMALIZATION.PERC_CLIP.ENABLE = False
        # Lower and upper bound for percentile clip. A float between 0 and 100.
        _C.DATA.NORMALIZATION.PERC_CLIP.LOWER_PERC = -1.0
        _C.DATA.NORMALIZATION.PERC_CLIP.UPPER_PERC = -1.0
        # Normalization methods. Options: "scale_range", 'zero_mean_unit_variance'.
        _C.DATA.NORMALIZATION.TYPE = "zero_mean_unit_variance"
        # 3.2. TRAIN
        _C.DATA.TRAIN = CN()
        # PATH organization
        # --- raw
        #   - xxx_0000.tif
        #   - xxx_0001.tif
        #   - xxx_0002.tif
        #   - ...
        _C.DATA.TRAIN.PATH = "./raw/"
        # PATH organization
        # --- label
        #   - xxx_gt_0000.tif
        #   - xxx_gt_0001.tif
        #   - xxx_gt_0002.tif
        #   - ...
        #   - xxx_gt_0000.[channel_name00].tif
        #   - xxx_gt_0001.[channel_name00].tif
        #   - xxx_gt_0002.[channel_name00].tif
        #   - ...
        #   - xxx_gt_0000.[channel_name01].tif
        #   - xxx_gt_0001.[channel_name01].tif
        #   - xxx_gt_0002.[channel_name01].tif
        #   - ...
        _C.DATA.TRAIN.GT_PATH = "./label/"
        # Percentage of overlap in (z,y,x) when cropping. Tuple of floats between range [0, 1).
        _C.DATA.TRAIN.OVERLAP = (0, 0, 0)
        # Padding to be done in (z,y,x). Useful to avoid patch 'border effect'. Tuples of ints.
        _C.DATA.TRAIN.PADDING = (0, 0, 0)
        # 3.3. VALIDATE
        _C.DATA.VAL = CN()
        _C.DATA.VAL.PATH = "./raw/"
        _C.DATA.VAL.GT_PATH = "./label/"
        # Percentage of overlap in (z,y,x) when cropping. Tuple of floats between range [0, 1).
        _C.DATA.VAL.OVERLAP = (0, 0, 0)
        # Padding to be done in (z,y,x). Useful to avoid patch 'border effect'. Tuples of ints.
        _C.DATA.VAL.PADDING = (0, 0, 0)
        # 3.3. INFER
        _C.DATA.INFER = CN()
        _C.DATA.INFER.PATH = "./raw/"
        _C.DATA.INFER.GT_PATH = "./label/"
        # Infer log
        _C.DATA.INFER.INFER_LOG = os.path.join(Path(_C.DATA.INFER.PATH).parent, "infer_log")
        # Whether to write results into original "hdf5"/"imaris"/"ome-zarr" files.
        _C.DATA.INFER.RESULT_INTO_FILE = True
        # Index of channel in "hdf5"/"imaris"/"ome-zarr" file used when inferring.
        _C.DATA.INFER.CHANNEL = [0]
        # Name of channel created in "hdf5"/"imaris"/"ome-zarr" file to save results.
        _C.DATA.INFER.CHANNEL_NAME = "result0"
        # Percentage of overlap in (z,y,x) when cropping. Tuple of floats between range [0, 1).
        _C.DATA.INFER.OVERLAP = (0, 0, 0)
        # Padding to be done in (z,y,x). Useful to avoid patch 'border effect'. Tuples of ints.
        _C.DATA.INFER.PADDING = (0, 0, 0)
        # Block used in inferring and watershed. Block size = BLOCK_FACTOR * PATCH_SIZE
        _C.DATA.INFER.BLOCK_FACTOR = (8, 6, 6)
        # Central part of block writen into results after inferring and watershed.
        _C.DATA.INFER.BLOCK_CENTRAL_FACTOR = (7, 5, 5)
        # Order of the axes of the image when using Zarr/hdf5 images.
        _C.DATA.INFER.INPUT_AXES_ORDER = "TCZYX"
        # 3.3.1 SKELETON
        _C.DATA.INFER.SKELETON_PATH = "./skeleton/"


        # 5. AUGMENTOR
        _C.AUGMENTOR = CN()
        # Flag to activate AUGMENTOR
        _C.AUGMENTOR.ENABLE = True
        # Probability of each transformation
        _C.AUGMENTOR.DA_PROB = 0.5
        # Flag to shuffle the training data on every epoch
        _C.AUGMENTOR.SHUFFLE_TRAIN_DATA_EACH_EPOCH = True
        # Flag to shuffle the validation data on every epoch
        _C.AUGMENTOR.SHUFFLE_VAL_DATA_EACH_EPOCH = False
        # Random rotation between a defined range
        _C.AUGMENTOR.RANDOM_ROT = False
        # Range of random rotations
        _C.AUGMENTOR.RANDOM_ROT_RANGE = (-180, 180)
        # Apply shear to images
        _C.AUGMENTOR.SHEAR = False
        # Shear range. Expected value range is around [-360, 360], with reasonable values being in the range of [-45, 45].
        _C.AUGMENTOR.SHEAR_RANGE = (-20, 20)
        # Apply zoom to images
        _C.AUGMENTOR.ZOOM = False
        # Zoom range. Scaling factor to use, where 1.0 denotes “no change” and 0.5 is zoomed out to 50 percent of the original size.
        _C.AUGMENTOR.ZOOM_RANGE = (0.9, 1.1)
        # Whether to apply or not zoom in Z axis (for 3D volumes).
        _C.AUGMENTOR.ZOOM_IN_Z = False
        # Apply shift
        _C.AUGMENTOR.SHIFT = False
        # Shift range. Translation as a fraction of the image height/width (x-translation, y-translation), where 0 denotes
        # “no change” and 0.5 denotes “half of the axis size”.
        _C.AUGMENTOR.SHIFT_RANGE = (0.1, 0.2)
        # How to fill up the new values created with affine transformations (rotations, shear, shift and zoom).
        # Only keep modes common to skimage & scipy: 'constant', 'reflect', 'wrap' and 'symmetric
        # Dropped 'edge'/'nearest' for simplicity
        _C.AUGMENTOR.AFFINE_MODE = "reflect"
        # Make vertical flips
        _C.AUGMENTOR.VFLIP = False
        # Make horizontal flips
        _C.AUGMENTOR.HFLIP = False
        # Make z-axis flips
        _C.AUGMENTOR.ZFLIP = False
        # Elastic transformations
        _C.AUGMENTOR.ELASTIC = False
        # Strength of the distortion field. Higher values mean that pixels are moved further with respect to the distortion
        # field's direction. Set this to around 10 times the value of sigma for visible effects.
        _C.AUGMENTOR.E_ALPHA = (12, 16)
        # Standard deviation of the gaussian kernel used to smooth the distortion fields.  Higher values (for 128x128 images
        # around 5.0) lead to more water-like effects, while lower values (for 128x128 images around 1.0 and lower) lead to
        # more noisy, pixelated images. Set this to around 1/10th of alpha for visible effects.
        _C.AUGMENTOR.E_SIGMA = 4
        # Parameter that defines the handling of newly created pixels with the elastic transformation
        _C.AUGMENTOR.E_MODE = "constant"
        # Gaussian blur
        _C.AUGMENTOR.G_BLUR = False
        # Standard deviation of the gaussian kernel. Values in the range 0.0 (no blur) to 3.0 (strong blur) are common.
        _C.AUGMENTOR.G_SIGMA = (1.0, 2.0)
        # To blur an image by computing median values over neighbourhoods
        _C.AUGMENTOR.MEDIAN_BLUR = False
        # Median blur kernel size
        _C.AUGMENTOR.MB_KERNEL = (3, 7)
        # Gamma contrast
        _C.AUGMENTOR.GAMMA_CONTRAST = False
        # Exponent for the contrast adjustment. Higher values darken the image
        _C.AUGMENTOR.GC_GAMMA = (0.8, 1.5)
        # To apply brightness changes to images
        _C.AUGMENTOR.BRIGHTNESS = False
        # Strength of the brightness range.
        _C.AUGMENTOR.BRIGHTNESS_FACTOR = (-0.05, 0.05)
        # To apply contrast changes to images
        _C.AUGMENTOR.CONTRAST = False
        # Strength of the contrast change range.
        _C.AUGMENTOR.CONTRAST_FACTOR = (-0.1, 0.1)
        # # To fill one or more rectangular areas in an image using a fill mode
        # _C.AUGMENTOR.CUTOUT = False
        # # Range of number of areas to fill the image with. Reasonable values between range [0,4]
        # _C.AUGMENTOR.COUT_NB_ITERATIONS = (1, 3)
        # # Size of the areas in % of the corresponding image size
        # _C.AUGMENTOR.COUT_SIZE = (0.05, 0.3)
        # # Value to fill the area of cutout
        # _C.AUGMENTOR.COUT_CVAL = 0.0
        # # Apply cutout to the segmentation mask
        # _C.AUGMENTOR.COUT_APPLY_TO_MASK = False
        # # To apply cutblur operation
        # _C.AUGMENTOR.CUTBLUR = False
        # # Size of the region to apply cutblur
        # _C.AUGMENTOR.CBLUR_SIZE = (0.2, 0.4)
        # # Range of the downsampling to be made in cutblur
        # _C.AUGMENTOR.CBLUR_DOWN_RANGE = (2, 8)
        # # Whether to apply cut-and-paste just LR into HR image. If False, HR to LR will be applied also (see Figure 1
        # # of the paper https://arxiv.org/pdf/2004.00448.pdf)
        # _C.AUGMENTOR.CBLUR_INSIDE = True
        # # Apply cutmix operation
        # _C.AUGMENTOR.CUTMIX = False
        # # Size of the region to apply cutmix
        # _C.AUGMENTOR.CMIX_SIZE = (0.2, 0.4)
        # # Apply noise to a region of the image
        # _C.AUGMENTOR.CUTNOISE = False
        # # Range to choose a value that will represent the % of the maximum value of the image that will be used as the std
        # # of the Gaussian Noise distribution
        # _C.AUGMENTOR.CNOISE_SCALE = (0.05, 0.1)
        # # Number of areas to fill with noise
        # _C.AUGMENTOR.CNOISE_NB_ITERATIONS = (1, 3)
        # # Size of the regions
        # _C.AUGMENTOR.CNOISE_SIZE = (0.2, 0.4)

        # 6. MODEL
        _C.MODEL = CN()
        # 5.1. ARCHITECTURE
        _C.MODEL.ARCHITECTURE = "unext_v2"
        # Number of feature maps on each level of the network.
        _C.MODEL.FEATURE_MAPS = [16, 32, 64, 128, 256]
        # Downsampling to be made in Z. When facing anysotropic datasets set it to get better performance.
        _C.MODEL.Z_DOWN = [1, 1, 2, 2]
        # Downsampling to be made in XY. When facing anysotropic datasets set it to get better performance.
        _C.MODEL.YX_DOWN = [2, 2, 2, 2]
        # For each level of the model (U-Net levels), set to true or false if the dimensions of the feature maps are isotropic.
        _C.MODEL.ISOTROPY = [True, True, True, True, True]
        # Number of ConvNeXtBlocks in each level.
        _C.MODEL.CONVNEXT_LAYERS = [2, 2, 2, 2, 2]  # CONVNEXT_LAYERS
        # Maximum Stochastic Depth probability for the U-NeXt model.
        _C.MODEL.CONVNEXT_SD_PROB = 0.1
        # Size of the stem kernel in the U-NeXt model.
        _C.MODEL.CONVNEXT_STEM_K_SIZE = 2
        # 5.2. CHECKPOINT
        _C.MODEL.LOAD_CHECKPOINT = False

        # 7. LOSS
        _C.LOSS = CN()

        # 8. TRAIN
        _C.TRAIN = CN()
        _C.TRAIN.ENABLE = False
        # Optimizer. Options: "SGD", "ADAM" or "ADAMW"
        _C.TRAIN.OPTIMIZER = "ADAMW"
        # Learning rate
        _C.TRAIN.LR = 5.E-4
        # Weight decay
        _C.TRAIN.W_DECAY = 0.02
        # Coefficients used for computing running averages of gradient and its square. Used in ADAM and ADAMW optmizers
        _C.TRAIN.OPT_BETAS = (0.9, 0.999)
        # Batch size
        _C.TRAIN.BATCH_SIZE = 2
        # Number of epochs to train the model
        _C.TRAIN.EPOCHS = 1000
        # Epochs to wait with no validation data improvement until the training is stopped
        _C.TRAIN.PATIENCE = 100
        # 7.1. LR Scheduler
        _C.TRAIN.LR_SCHEDULER = CN()
        # Ooptions: "warmupcosine", "onecycle"
        _C.TRAIN.LR_SCHEDULER.NAME = "warmupcosine"
        # Lower bound on the learning rate used in "warmupcosine"
        _C.TRAIN.LR_SCHEDULER.MIN_LR = 1.E-6
        # Epochs to do the warming up in "warmupcosine".
        _C.TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_EPOCHS = 5

        # 9. INFERENCE
        _C.INFER = CN()
        _C.INFER.ENABLE = False
        # 8.1. SKELETON
        _C.INFER.SKELETON = CN()
        # Whether to use .swc skeleton files as seed channel when predicting.
        # Default: True.
        _C.INFER.SKELETON.ENABLE = True
        # 8.2.
        # Whether to save the raw output of the model (before any post-processing). Set to True when testing model.
        _C.INFER.SAVE_MODEL_RAW_OUTPUT = False
        # When SKELETON.ENAMBLE is True, set as 0. Use -1 (default) to compute it automatically as patch_size / 8.
        _C.INFER.INSTANCE_SEG_HALO = -1
        # 8.3. POST-PROCESSING
        _C.INFER.POST_PROCESSING = CN()
        _C.INFER.POST_PROCESSING.ENABLE = False
        # Options: "dilation", "erosion", "fill_holes", "remove_small", "remove_large".
        _C.INFER.POST_PROCESSING.OPERATIONS = ["fill_holes", "remove_small"]
        _C.INFER.POST_PROCESSING.VALUES = [None, 100]

        self._C = _C

    def get_cfg_defaults(self) -> CN:
        return self._C.clone()

    def to_dict(self):
        return dict(self._C)

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


def load_config(config:str) ->Dict:
    """
    Load dict from yaml config file.
    :param
    config: str, yaml file path
    :return:
    """
    config_path = Path(config)
    assert config_path.exists(), "Config file does not exist."
    assert config_path.suffix in [".yaml", ".yml"], "Config file extension must be .yaml or .yml."
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = CN.load_cfg(f)
    assert isinstance(cfg, CN), "Config file type is not CN."
    return cfg


def update_config(cfg:CN, new_cfg:CN, job_id:Optional[str|int]) -> CN:
    assert isinstance(cfg, CN) and isinstance(new_cfg, CN), "Config type is not CN."
    if cfg.is_frozen():
        cfg.defrost()

    def _update_recursive(base_cfg: CN, override_cfg: CN, prefix: str = ""):
        for key, value in override_cfg.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if key not in base_cfg:
                raise KeyError(
                    f"Unknown config key: {full_key}. "
                    f"Please check whether this key exists in default Config."
                )

            base_value = base_cfg[key]

            if isinstance(base_value, CN) and isinstance(value, CN):
                _update_recursive(base_value, value, full_key)
            else:
                base_cfg[key] = copy.deepcopy(value)
    _update_recursive(cfg, new_cfg)
    cfg.PATHS.RESULT_DIR.PATH_ = os.path.join(cfg.PATHS.RESULT_DIR.PATH, "results", str(job_id))
    cfg.PATHS.RESULT_DIR.OUTPUT_LOG = os.path.join(cfg.PATHS.RESULT_DIR.PATH_, "log")
    cfg.DATA.INFER.INFER_LOG = os.path.join(Path(cfg.DATA.INFER.PATH).parent, "infer_log")
    cfg.PATHS.RESULT_DIR.OUTPUT_CHART = os.path.join(cfg.PATHS.RESULT_DIR.PATH_, "chart")
    cfg.PATHS.RESULT_DIR.OUTPUT_RAW = os.path.join(cfg.PATHS.RESULT_DIR.PATH_, "output_raw")
    cfg.PATHS.RESULT_DIR.OUTPUT_INSTANCES = os.path.join(cfg.PATHS.RESULT_DIR.PATH_, "output_instances")
    cfg.PATHS.RESULT_DIR.OUTPUT_POST_PROCESSING = os.path.join(cfg.PATHS.RESULT_DIR.PATH_, "output_post_processing")
    cfg.PATHS.CHECKPOINT_DIR = os.path.join(cfg.PATHS.RESULT_DIR.PATH_, "checkpoints")
    return cfg



