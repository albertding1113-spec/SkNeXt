import numpy as np
from pathlib import Path
import navis
import json
import tifffile
from scipy import ndimage as ndi
from sknext.skeleton.skeleton import SkeletonManager
from sknext.utils.progress import time_str
from sknext.data.imageIO import get_coord_after_padding, central_block_coordinates_iter

class Skeleton_Workflow():
    """Extract segmented components associated with neuronal skeletons.

    Processes volume blocks, filters objects by size/location and skeleton
    compartment, and exports component masks with source/path metadata.
    """
    def __init__(self,
                 skeleton_path,
                 file_path,
                 result_path,
                 channel = (0, 1),
                 central_block = (256, 1024, 1024),
                 block_halo = (64, 128, 128),
                 connectivity = 1,
                 min_values = (10, 10),
                 subregion = ((0, 1, 2, 3, 4), (3, 4))):
        """
        :param subregion: 0 undefined, 1 soma, 2 axon, 3 basal dendrite, 4 apical dendrite
                    Example-- mito: [0,1,2,3,4] or [-1], spine: [3,4], bouton: [2]. [-1] for other cell types.
        """
        self.skeleton_path = Path(skeleton_path)
        self.file_path = Path(file_path)
        self.result_path = Path(result_path)
        self.channel = tuple(channel)
        self.ch_num = len(channel)
        self.central_block = tuple(central_block)
        self.block_halo = tuple(block_halo)
        self.connectivity = connectivity
        self.min_values = tuple(min_values)
        self.subregion = tuple(tuple(labels) for labels in subregion)
        self._validate_settings()
        self.structure = ndi.generate_binary_structure(3, self.connectivity)

        self.create_skeleton()
        self.create_reader()
        try:
            if len(self.reader.shape) != 4 or any(n <= 0 for n in self.reader.shape):
                raise ValueError("Input must have a nonempty CZYX shape.")
            if any(ch >= self.reader.shape[0] for ch in self.channel):
                raise ValueError("Selected channel is outside the input CZYX shape.")
            self.create_result_dir()
        except Exception:
            self.reader.close()
            raise

    def _validate_settings(self):
        def integers(values, minimum):
            return all(isinstance(v, (int, np.integer))
                       and not isinstance(v, (bool, np.bool_))
                       and v >= minimum for v in values)

        if not self.channel or not integers(self.channel, 0):
            raise ValueError("channel must contain non-negative integer indices.")
        if len(set(self.channel)) != self.ch_num:
            raise ValueError("channel must not contain duplicates.")
        for name, values, minimum in (("central_block", self.central_block, 1),
                                      ("block_halo", self.block_halo, 0)):
            if len(values) != 3 or not integers(values, minimum):
                raise ValueError(f"{name} must contain three integers >= {minimum} in ZYX order.")
        if not integers((self.connectivity,), 1) or self.connectivity > 3:
            raise ValueError("connectivity must be 1, 2, or 3.")
        if len(self.min_values) != self.ch_num or not integers(self.min_values, 1):
            raise ValueError("min_values must contain one positive integer per selected channel.")
        if len(self.subregion) != self.ch_num or any(
                not labels or not integers(labels, -1) for labels in self.subregion):
            raise ValueError("subregion must contain one nonempty integer label sequence per channel.")

    def create_skeleton(self):
        """Load and prepare the configured SWC directory as self.skeleton."""
        self.skeleton = SkeletonManager(self.skeleton_path)

    def create_reader(self):
        """Open the configured volume as an OME-Zarr or Imaris reader.

        Stores self.reader; unsupported input formats raise FileNotFoundError.
        """
        from sknext.data.datasetIO import detect_path_type, IMSReader, ZarrIOManager

        file_type = detect_path_type(self.file_path)
        if file_type == "zarr":
            self.reader = ZarrIOManager(self.file_path, mode="r")
        elif file_type == "ims":
            self.reader = IMSReader(self.file_path)
        else:
            raise FileNotFoundError(f"file type {file_type} is not supported.")

    def create_result_dir(self):
        """Create the result root and one output directory per selected channel."""
        self.result_dirs = []
        self.result_path.mkdir(parents=True, exist_ok=True)
        for ch in self.channel:
            self.result_dirs.append(self.result_path / str(ch))
            self.result_path.joinpath(str(ch)).mkdir(parents=True, exist_ok=True)

    def calc_save_one_block(self, block, skeleton_mask, ch, coord, central_coord_in_block):
        """Filter connected components in one channel block and export accepted objects.

        Args:
            block: ZYX segmentation block; values at least one are foreground.
            skeleton_mask: Aligned mask whose positive IDs are skeleton indices plus one.
            ch: Position in the selected channel list and its per-channel settings.
            coord: Global (3, 2) ZYX bounds of the block.
            central_coord_in_block: Block-local (3, 2) bounds used to assign components
                uniquely by their centroid.

        Rejects small, boundary-touching, unseeded, or disallowed-compartment
        objects. Writes JSON bounds, a binary TIFF, and an SWC path to soma/root.
        """

        def get_largest_id(mask: np.ndarray, ignore_ids: tuple[int, ...] = (0,),) -> int:
            """Return the most frequent ID in mask excluding ignore_ids.

            Returns zero when no eligible values remain; ties follow sorted ID order.
            """
            if np.count_nonzero(mask) == 0:
                return 0
            valid_mask = ~np.isin(mask, ignore_ids)
            valid_values = mask[valid_mask]
            if valid_values.size == 0:
                return 0
            ids, counts = np.unique(valid_values, return_counts=True)
            largest_id = ids[np.argmax(counts)]
            return int(largest_id)

        if block.ndim != 3 or skeleton_mask.shape != block.shape:
            raise ValueError("block and skeleton_mask must have matching ZYX shapes.")
        coord = np.asarray(coord)
        central_coord_in_block = np.asarray(central_coord_in_block)
        component_labels, component_num = ndi.label(block >= 1, structure=self.structure)
        if component_num == 0: return
        # Only components intersecting a skeleton can be exported. Avoid
        # constructing masks and measuring centroids for all unseeded objects.
        candidates = np.unique(component_labels[skeleton_mask > 0])
        candidates = candidates[candidates > 0]
        if not candidates.size:
            return
        component_slices = ndi.find_objects(component_labels)
        for comp_id in candidates:
            comp_slice = component_slices[comp_id - 1]
            # Reject possibly truncated objects before allocating their masks.
            if any(s.start == 0 or s.stop == block.shape[i]
                   for i, s in enumerate(comp_slice)):
                continue
            one_comp = component_labels[comp_slice] == comp_id
            # 1. voxel_count >= min_size
            voxel_count = int(np.count_nonzero(one_comp))
            if voxel_count < self.min_values[ch]: continue
            # 2. locate in central_block
            centroid_zyx = ndi.center_of_mass(one_comp)
            comp_center = [comp_slice[i].start + centroid_zyx[i] for i in range(3)]
            if not np.all([central_coord_in_block[i,0] <= comp_center[i] < central_coord_in_block[i,1] for i in range(3)]):
                continue
            # 4. has overlap with skeleton mask
            one_sk_mask = skeleton_mask[comp_slice]
            overlapped_mask = one_sk_mask[one_comp]
            sk_idx = get_largest_id(overlapped_mask)
            if sk_idx == 0: continue
            sk_idx = sk_idx - 1
            # 5. match with nearest node
            r_comp_center = np.asarray(comp_center) + coord[:, 0]
            nearest_node = self.skeleton.get_nearest_skeleton_node(r_comp_center, sk_idx) # navis.TreeNeuron
            if (nearest_node["label"] not in self.subregion[ch]) and (-1 not in self.subregion[ch]): continue
            # save sk_id, min_coord, tif image, swc file
            min_coord = [[int(coord[i,0] + comp_slice[i].start), int(coord[i,0] + comp_slice[i].stop)] for i in range(3)]
            saved_json = {"sk_idx": sk_idx, "min_coord": min_coord,
                          "channel": self.channel[ch], "axes": "ZYX",
                          "centroid_zyx": r_comp_center.tolist(),
                          "voxel_count": voxel_count,
                          "swc_filename": self.skeleton.swc_paths[sk_idx].name,
                          "swc_file_id": self.skeleton.file_ids[sk_idx]}
            path_to_soma = self.skeleton.get_path_to_soma_or_root(sk_idx, r_comp_center, nearest_node)
            saved_path = self.result_dirs[ch]/f"{sk_idx:05d}"/f"z{min_coord[0][0]:07d}_y{min_coord[1][0]:07d}_x{min_coord[2][0]:07d}"
            # Disjoint components can have identical bounding-box minima.
            # Their first occupied voxel is unique and stable across blocks.
            first = np.asarray(np.unravel_index(np.argmax(one_comp), one_comp.shape))
            first += np.asarray(min_coord)[:, 0]
            saved_path = saved_path.with_name(
                saved_path.name + f"_seed_z{first[0]}_y{first[1]}_x{first[2]}")
            saved_path.mkdir(parents=True, exist_ok=True)
            with open(saved_path/"data.json", "w", encoding="utf-8") as f:
                json.dump(saved_json, f, ensure_ascii=False, indent=4)
            tifffile.imwrite(saved_path/"data.tif", one_comp.astype('uint8'),
                             compression='deflate', photometric='minisblack',
                             metadata={"axes": "ZYX"})
            # NAVis initializes its output 'label' column to zero before
            # selecting the requested column. Preserve labels under a separate
            # name so labels="label" cannot silently erase SWC compartments.
            path_to_soma.nodes["export_label"] = path_to_soma.nodes["label"].copy()
            navis.write_swc(path_to_soma, saved_path/"data.swc", labels="export_label")

    def run(self):
        """Process spatial blocks and selected channels, exporting skeleton-linked objects.

        Skips blocks without skeletons, prints progress/errors, and closes the
        reader in a finally block. Processing exceptions propagate to the caller.
        """
        try:
            # calculate central_block coordinates
            spatial_shape = self.reader.shape[1:4]
            central_coord = central_block_coordinates_iter(spatial_shape, self.central_block)
            total_block_num = int(np.prod([
                (n + b - 1) // b for n, b in zip(spatial_shape, self.central_block)]))
            # start calculate
            for i, c_coord in enumerate(central_coord):
                coord, coord_in_block = get_coord_after_padding(self.reader.shape[1:4], c_coord, self.block_halo)
                cropped_skeleton = self.skeleton.crop_skeletons(coord) # navis.NeuronList
                if len(cropped_skeleton) == 0: continue
                skeleton_mask = self.skeleton.create_cropped_skeleton_mask(cropped_skeleton, coord)
                for ch in range(len(self.channel)):
                    block = self.reader[self.channel[ch], coord[0, 0]:coord[0, 1], coord[1, 0]:coord[1, 1], coord[2, 0]:coord[2, 1],]
                    self.calc_save_one_block(block, skeleton_mask, ch, coord, coord_in_block)
                    print(f"{time_str()} [BLOCK{i + 1:07d}/{total_block_num:07d}] "
                          f"[CHANNEL {self.channel[ch]}] [SKELETON] processed", flush=True)
        finally:
            self.reader.close()

if __name__ == "__main__":
    sk = Skeleton_Workflow(
                     skeleton_path = r"/HWFS/zhaohu_lab/dingjiayi/data/260601_TEEX_whole_brain/260822_x10000-38000y65000-120000z7000-16000/skeleton_whole",
                     file_path = r"/HWFS/zhaohu_lab/dingjiayi/data/260601_TEEX_whole_brain/260822_x10000-38000y65000-120000z7000-16000/channel1/260822_x10000-38000y65000-120000z7000-16000_channel1.ome.zarr",
                     result_path = r"/HWFS/zhaohu_lab/dingjiayi/data/260601_TEEX_whole_brain/260822_x10000-38000y65000-120000z7000-16000/mito_along_axon",
                     channel=[0],
                     min_values=[10],
                     subregion=[[2]])
    sk.run()
