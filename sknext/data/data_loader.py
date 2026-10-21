import torch
import numpy as np
from pathlib import Path
import zarr
import random
from zarr.storage import LocalStore
from sknext.data.augmentor import PairedAugmentor
from sknext.utils.utils import get_cfg_value


class ZarrPatchLoader:
    """Read paired NCZYX patch datasets and assemble batches for segmentation.

    Applies configured random augmentation to training samples only, while
    validation reads retain the stored raw/label pairing.
    """
    def __init__(self, cfg,
                 zarr_path: str | Path,
                 dataset_name: list[str] | tuple[str] = ("raw", "label"),
                 if_val: bool = False):
        """Open a patch store and cache augmentation options from cfg.

        Args:
            cfg: Configuration supplying AUGMENTOR options.
            zarr_path: Existing Zarr group containing paired patch arrays.
            dataset_name: Raw and label array names, in that order.
            if_val: Disable training augmentation for validation reads.

        Raises:
            KeyError: A requested dataset is absent from the store.
        """
        self.cfg = cfg
        self.zarr_path = Path(zarr_path)
        assert self.zarr_path.exists(), "zarr_path does not exist."
        self.store = LocalStore(str(self.zarr_path))
        self.root = zarr.open_group(store=self.store, mode="r")
        self.dataset_name = dataset_name
        self.if_val = if_val
        self.arrays = {}
        for name in self.dataset_name:
            if name not in self.root:
                raise KeyError(f"Dataset '{name}' not found in zarr file. Available datasets: {list(self.root.array_keys())}")
            arr = self.root[name]
            self.arrays[name] = arr
        self.num_patches = self.arrays[dataset_name[0]].shape[0]
        self.augmentor_flag = bool(get_cfg_value(self.cfg, "AUGMENTOR.ENABLE", True))
        self.augmentor = PairedAugmentor(self.cfg, task="segmentation")

    def __len__(self) -> int:
        """Return the number of stored patches, rather than the number of batches."""
        return self.num_patches

    def load_one_pair_patch(self, id:int) -> tuple[np.ndarray, np.ndarray]:
        """Read a raw/label pair at the zero-based patch id.

        Returns:
            Two CZYX NumPy arrays, augmented together for training when enabled.

        Raises:
            IndexError: id is outside the stored patch range.
        """
        if id < 0 or id >= self.num_patches:
            raise IndexError(f"Patch id out of range: {id}, valid range is [0, {self.num_patches - 1}]")
        raw_arr = self.arrays[self.dataset_name[0]]
        raw_patch = np.asarray(raw_arr[id, :, :, :, :])
        raw_patch = np.transpose(raw_patch, (1, 2, 3, 0))  # CZYX -> ZYXC
        label_arr = self.arrays[self.dataset_name[1]]
        label_patch = np.asarray(label_arr[id, :, :, :, :])
        label_patch = np.transpose(label_patch, (1, 2, 3, 0))   # CZYX -> ZYXC
        if self.augmentor_flag and not self.if_val:
            raw_patch, label_patch = self.patch_augment(raw_patch, label_patch)
        raw_patch = np.transpose(raw_patch, (3, 0, 1, 2))  # ZYXC -> CZYX
        label_patch = np.transpose(label_patch, (3, 0, 1, 2))  # ZYXC -> CZYX
        return raw_patch, label_patch

    def patch_augment(self, patch: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply the shared paired augmentation policy to a ZYXC patch."""
        return self.augmentor(patch, mask)

    def get_epoch_indices(self, shuffle: bool = False) -> np.ndarray:
        """Return patch indices for one complete epoch.

        Parameters
        ----------
        shuffle : bool, default=False
            If True, return a new random permutation of all patch indices.
            Otherwise, return indices in their original Zarr order.

        Notes
        -----
        Each patch appears exactly once in the returned array, regardless of
        whether shuffling is enabled.
        """
        if shuffle:
            return np.random.permutation(self.num_patches).astype(np.int64, copy=False)
        return np.arange(self.num_patches, dtype=np.int64)

    def _load_index_batch(self, indices) -> tuple[torch.Tensor, torch.Tensor]:
        """Load an arbitrary collection of patch indices as one tensor batch."""
        indices = np.asarray(indices)
        if indices.ndim != 1:
            raise IndexError(f"Batch indices must be one-dimensional, got shape {indices.shape}.")
        if indices.size == 0:
            raise IndexError("Batch indices must not be empty.")
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError(f"Batch indices must be integers, got dtype {indices.dtype}.")

        imgs = []
        masks = []
        for i in indices:
            img, mask = self.load_one_pair_patch(int(i))
            imgs.append(torch.from_numpy(img.astype("float32", copy=False)))
            masks.append(torch.from_numpy(mask.astype("float32", copy=False)))

        return torch.stack(imgs, dim=0), torch.stack(masks, dim=0)

    def __getitem__(
        self,
        id: int | np.integer | slice | list[int] | tuple[int, ...] | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one patch or an arbitrary batch of patches.

        Integer indexing returns one CZYX image/label pair.
        Slice/list/tuple/ndarray indexing returns a batch in NCZYX order.
        """
        if isinstance(id, (int, np.integer)):
            img, mask = self.load_one_pair_patch(int(id))
            img = torch.from_numpy(img.astype("float32", copy=False))
            mask = torch.from_numpy(mask.astype("float32", copy=False))
            return img, mask

        if isinstance(id, slice):
            indices = np.arange(self.num_patches, dtype=np.int64)[id]
            return self._load_index_batch(indices)

        if isinstance(id, (list, tuple, np.ndarray)):
            return self._load_index_batch(id)

        raise TypeError(
            "id must be an int, slice, list, tuple, or one-dimensional integer ndarray, "
            f"got {type(id).__name__}."
        )
