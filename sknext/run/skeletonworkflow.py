import numpy as np
import pandas as pd
import navis
import json
import tifffile
from scipy import ndimage as ndi
from sknext.skeleton.skeleton import *
from sknext.data.datasetIO import detect_path_type, IMSReader, ZarrIOManager
from sknext.utils.utils import time_str
from sknext.data.imageIO import get_coord_after_padding, central_block_coordinates_iter

class Skeleton_Workflow():
    def __init__(self,
                 skeleton_path,
                 file_path,
                 result_path,
                 channel = [0,1],
                 central_block = (256, 1024, 1024),
                 block_halo = (32, 128, 128),
                 connectivity = 1,
                 min_values = [10,10],
                 subregion = [[0,1,2,3,4], [3,4]]):
        """
        :param subregion: 0 undefined, 1 soma, 2 axon, 3 basal dendrite, 4 apical dendrite
                    Example-- mito: [0,1,2,3,4] or [-1], spine: [3,4], bouton: [2]. [-1] for other cell types.
        """
        self.skeleton_path = Path(skeleton_path)
        self.file_path = Path(file_path)
        self.result_path = Path(result_path)
        self.channel = channel
        self.ch_num = len(channel)
        self.central_block = central_block
        self.block_halo = block_halo
        self.connectivity = connectivity
        self.min_values = min_values
        self.subregion = subregion

        self.create_skeleton()
        self.create_reader()
        self.create_result_dir()

    def create_skeleton(self):
        self.skeleton = SkeletonManager(self.skeleton_path)

    def create_reader(self):
        file_type = detect_path_type(self.file_path)
        if file_type == "zarr":
            self.reader = ZarrIOManager(self.file_path, mode="r")
        elif file_type == "ims":
            self.reader = IMSReader(self.file_path)
        else:
            raise FileNotFoundError(f"file type {file_type} is not supported.")

    def create_result_dir(self):
        self.result_dirs = []
        self.result_path.mkdir(parents=True, exist_ok=True)
        for ch in self.channel:
            self.result_dirs.append(self.result_path / str(ch))
            self.result_path.joinpath(str(ch)).mkdir(parents=True, exist_ok=True)

    def calc_save_one_block(self, block, skeleton_mask, ch, coord, central_coord_in_block):

        def get_largest_id(mask: np.ndarray, ignore_ids: tuple[int, ...] = (0,),) -> int:
            if np.count_nonzero(mask) == 0:
                return 0
            valid_mask = ~np.isin(mask, ignore_ids)
            valid_values = mask[valid_mask]
            if valid_values.size == 0:
                return 0
            ids, counts = np.unique(valid_values, return_counts=True)
            largest_id = ids[np.argmax(counts)]
            return int(largest_id)

        block_mask = np.uint8(block >= 1)
        structure = ndi.generate_binary_structure(rank=3, connectivity=int(self.connectivity),)
        component_labels, component_num = ndi.label(block_mask, structure=structure,)
        if component_num == 0: return
        component_slices = ndi.find_objects(component_labels)
        for comp_id, comp_slice in enumerate(component_slices, start=1,):
            if comp_slice is None: continue
            one_comp = component_labels[comp_slice] == comp_id
            # 1. voxel_count >= min_size
            voxel_count = int(np.count_nonzero(one_comp))
            if voxel_count < self.min_values[ch]: continue
            # 2. locate in central_block
            centroid_zyx = ndi.center_of_mass(one_comp)
            comp_center = [comp_slice[i].start + centroid_zyx[i] for i in range(3)]
            if not np.all([central_coord_in_block[i,0] <= comp_center[i] < central_coord_in_block[i,1] for i in range(3)]):
                continue
            # 3. not connect to boundary
            if np.any(np.array([[comp_slice[i].start == 0, comp_slice[i].stop == block.shape[i]] for i in range(3)])):
                continue
            # 4. has overlap with skeleton mask
            one_sk_mask = skeleton_mask[comp_slice]
            overlapped_mask = one_sk_mask[one_comp]
            sk_idx = get_largest_id(overlapped_mask)
            if sk_idx == 0: continue
            sk_idx = sk_idx - 1
            # 5. match with nearest node
            r_comp_center = comp_center + coord[:, 0]
            nearest_node = self.skeleton.get_nearest_skeleton_node(r_comp_center, sk_idx) # navis.TreeNeuron
            if (nearest_node["label"] not in self.subregion[ch]) and (-1 not in self.subregion[ch]): continue
            # save sk_id, min_coord, tif image, swc file
            min_coord = [[int(coord[i,0] + comp_slice[i].start), int(coord[i,0] + comp_slice[i].stop)] for i in range(3)]
            saved_json = {"sk_idx": sk_idx, "min_coord": min_coord,}
            path_to_soma = self.skeleton.get_path_to_soma_or_root(sk_idx, r_comp_center, nearest_node)
            saved_path = self.result_dirs[ch]/str(sk_idx)/f"z{min_coord[0][0]:07d}_y{min_coord[1][0]:07d}_x{min_coord[2][0]:07d}"
            saved_path.mkdir(parents=True, exist_ok=True)
            with open(saved_path/"data.json", "w", encoding="utf-8") as f:
                json.dump(saved_json, f, ensure_ascii=False, indent=4)
            tifffile.imwrite(saved_path/"data.tif", one_comp, compression='lzw')
            navis.write_swc(path_to_soma, saved_path/"data.swc", labels="label")

    def run(self):
        try:
            # calculate central_block coordinates
            central_coord = list(central_block_coordinates_iter(self.reader.shape[1:4], self.central_block))
            total_block_num = len(central_coord)
            # start calculate
            for i, c_coord in enumerate(central_coord):
                coord, coord_in_block = get_coord_after_padding(self.reader.shape[1:4], c_coord, self.block_halo)
                cropped_skeleton = self.skeleton.crop_skeletons(coord) # navis.NeuronList
                if len(cropped_skeleton) == 0: continue
                skeleton_mask = self.skeleton.create_cropped_skeleton_mask(cropped_skeleton, coord)
                for ch in range(len(self.channel)):
                    block = self.reader[self.channel[ch], coord[0, 0]:coord[0, 1], coord[1, 0]:coord[1, 1], coord[2, 0]:coord[2, 1],]
                    self.calc_save_one_block(block, skeleton_mask, ch, coord, coord_in_block)
                    print(f"{time_str()} [BLOCK{i:07d}/{total_block_num:07d}] [SKELETON] results saved", flush=True)
        except Exception as e:
            print(f"{e}")
        finally:
            self.reader.close()

if __name__ == "__main__":
    sk = Skeleton_Workflow(
                     skeleton_path = r"G:\Albert\data\260618_SkNeXt_dataset\skeleton2",
                     file_path = r"G:\260501_250708_fTEEX_region2_test2.ome.zarr",
                     result_path = r"G:\bouton",
                     channel=[3],
                     min_values=[30],
                     subregion=[[2]])
    sk.run()