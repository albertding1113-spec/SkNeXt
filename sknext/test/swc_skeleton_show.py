import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_swc(swc_path, voxel_size_xyz=None):
    """
    Load an SWC file and optionally convert voxel coordinates
    to physical coordinates.

    SWC columns:
        0: node id
        1: type
        2: x
        3: y
        4: z
        5: radius
        6: parent id

    Parameters
    ----------
    swc_path : str
        Path to SWC file.

    voxel_size_xyz : tuple or list, optional
        Physical voxel size in XYZ order.

        Example:
            voxel_size_xyz = (0.15, 0.15, 0.5)

        If SWC coordinates are voxel indices, this converts them
        to physical coordinates.

        If None, original SWC coordinates are preserved.
    """

    data = np.loadtxt(swc_path, comments="#")

    node_id = data[:, 0].astype(int)
    node_type = data[:, 1].astype(int)

    xyz = data[:, 2:5].astype(float)

    radius = data[:, 5].astype(float)
    parent_id = data[:, 6].astype(int)

    # ---------------------------------------------------------
    # Convert voxel coordinates -> physical coordinates
    # ---------------------------------------------------------
    if voxel_size_xyz is not None:

        voxel_size_xyz = np.asarray(
            voxel_size_xyz,
            dtype=float
        )

        if voxel_size_xyz.shape != (3,):
            raise ValueError(
                "voxel_size_xyz must contain exactly 3 values: "
                "(X, Y, Z)"
            )

        xyz = xyz * voxel_size_xyz

    return node_id, node_type, xyz, radius, parent_id


def show_swc(
    swc_path,
    voxel_size_xyz=None,
    soma_color="red",
    axon_color="blue",
    dendrite_color="green",
    other_color="gray",
    linewidth=1.5,
    soma_size=30,
    show_axis=True,
):
    """
    Visualize an SWC skeleton in 3D.

    Standard SWC types:
        1 = soma
        2 = axon
        3 = basal dendrite
        4 = apical dendrite

    Parameters
    ----------
    swc_path : str
        Path to SWC file.

    voxel_size_xyz : tuple, optional
        Physical voxel size in XYZ order.

        Example:
            (0.15, 0.15, 0.5)

        If None, original SWC coordinates are used.
    """

    node_id, node_type, xyz, radius, parent_id = load_swc(
        swc_path,
        voxel_size_xyz=voxel_size_xyz
    )

    # Map SWC node ID -> array index
    id_to_index = {
        nid: i for i, nid in enumerate(node_id)
    }

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # ---------------------------------------------------------
    # Draw skeleton edges
    # ---------------------------------------------------------
    for i in range(len(node_id)):

        parent = parent_id[i]

        # Root node
        if parent == -1:
            continue

        if parent not in id_to_index:
            continue

        parent_idx = id_to_index[parent]

        p1 = xyz[parent_idx]
        p2 = xyz[i]

        # Color according to child node type
        t = node_type[i]

        if t == 1:
            color = soma_color

        elif t == 2:
            color = axon_color

        elif t in (3, 4):
            color = dendrite_color

        else:
            color = other_color

        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color=color,
            linewidth=linewidth,
        )

    # ---------------------------------------------------------
    # Draw soma
    # ---------------------------------------------------------
    soma_mask = node_type == 1

    if np.any(soma_mask):

        soma_xyz = xyz[soma_mask]

        ax.scatter(
            soma_xyz[:, 0],
            soma_xyz[:, 1],
            soma_xyz[:, 2],
            color=soma_color,
            s=soma_size,
            depthshade=False,
        )

    # ---------------------------------------------------------
    # Axis
    # ---------------------------------------------------------
    if voxel_size_xyz is not None:
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.set_zlabel("Z (µm)")
    else:
        ax.set_xlabel("X (voxel)")
        ax.set_ylabel("Y (voxel)")
        ax.set_zlabel("Z (voxel)")

    ax.set_title("SWC neuronal skeleton")

    # Physically correct XYZ aspect ratio
    x_range = np.ptp(xyz[:, 0])
    y_range = np.ptp(xyz[:, 1])
    z_range = np.ptp(xyz[:, 2])

    ax.set_box_aspect(
        (
            max(x_range, 1e-6),
            max(y_range, 1e-6),
            max(z_range, 1e-6),
        )
    )

    # ---------------------------------------------------------
    # Legend
    # ---------------------------------------------------------
    legend_elements = [
        Line2D(
            [0], [0],
            color=soma_color,
            lw=3,
            label="Soma",
        ),
        Line2D(
            [0], [0],
            color=axon_color,
            lw=3,
            label="Axon",
        ),
        Line2D(
            [0], [0],
            color=dendrite_color,
            lw=3,
            label="Dendrite",
        ),
    ]

    ax.legend(handles=legend_elements)

    if not show_axis:
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    swc_path = r"G:\Albert\data\260618_SkNeXt_dataset\skeleton2\001.swc"
    show_swc(
        swc_path,
        voxel_size_xyz=(0.149, 0.149, 0.5),
        linewidth=1.0,
        soma_size=50,
        show_axis=True,
    )