import numpy as np
import tifffile
from scipy import ndimage
from skimage.morphology import skeletonize


def instance_mask_to_skeleton(
    instance_mask,
    preserve_ids=True,
    output_dtype=np.uint32,
    padding=1,
):
    """
    Skeletonize each instance independently.

    Parameters
    ----------
    instance_mask : np.ndarray
        2D or 3D instance mask.
        0 = background
        1, 2, 3, ... = instance IDs

    preserve_ids : bool, default=True
        If True:
            skeleton voxels preserve their original instance IDs.

            Example:
                instance 15 -> skeleton value 15

        If False:
            all skeleton voxels are written as 1.

            Output:
                0 = background
                1 = skeleton

    output_dtype : numpy dtype
        Output array dtype.

    padding : int
        Padding added around each instance bounding box.

    Returns
    -------
    skeleton_mask : np.ndarray
        Skeleton array with the same shape as `instance_mask`.
    """

    if instance_mask.ndim not in (2, 3):
        raise ValueError(
            f"Only 2D/3D masks are supported. Got shape {instance_mask.shape}"
        )

    skeleton_mask = np.zeros(
        instance_mask.shape,
        dtype=output_dtype
    )

    max_id = int(instance_mask.max())

    # Bounding boxes for each label
    objects = ndimage.find_objects(
        instance_mask,
        max_label=max_id
    )

    for instance_id, bbox in enumerate(objects, start=1):

        # This instance ID does not exist
        if bbox is None:
            continue

        # Add padding
        padded_bbox = []

        for axis, s in enumerate(bbox):
            start = max(0, s.start - padding)
            stop = min(
                instance_mask.shape[axis],
                s.stop + padding
            )

            padded_bbox.append(
                slice(start, stop)
            )

        padded_bbox = tuple(padded_bbox)

        # Extract local region
        local_labels = instance_mask[padded_bbox]

        # Binary mask of current instance
        local_instance = (
            local_labels == instance_id
        )

        if not np.any(local_instance):
            continue

        # Skeletonize
        local_skeleton = skeletonize(
            local_instance
        )

        # Write back
        output_view = skeleton_mask[padded_bbox]

        if preserve_ids:
            output_view[local_skeleton] = instance_id
        else:
            output_view[local_skeleton] = 1

    return skeleton_mask


if __name__ == "__main__":
    instance_mask = tifffile.imread(r'G:\Albert\data\260424_iterative_proofreading_training\raw\0013\0013_gt.tif')
    skeleton_mask = instance_mask_to_skeleton(instance_mask, preserve_ids=False, output_dtype=np.uint8)
    tifffile.imwrite(r'G:\Albert\data\260424_iterative_proofreading_training\raw\0013\0013_skeleton.tif', skeleton_mask)
