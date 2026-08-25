from __future__ import annotations
import sys
import navis
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Iterable
from collections.abc import Sequence
import tifffile


class SkeletonManager():
    def __init__(self,
                 sk_path: str | Path,
                 node_distance: float | Iterable[float] = [3,10,10],):
        self.path = Path(sk_path)
        assert self.path.exists() and self.path.is_dir(), "skeleton path does not exist"
        self.skeletons = navis.read_swc(sk_path)
        assert len(self.skeletons) > 0, "skeleton files do not exist"
        self.skeletons = self.fix_skeletons_root(self.skeletons)
        self._build_id_index_dict()
        self.increase_nodes_density(node_distance)
        self.increase_radius()

    def __len__(self):
        return len(self.skeletons)

    def __getitem__(self, idx):
        return self.skeletons[idx]

    def __iter__(self):
        return iter(self.skeletons)

    def _build_id_index_dict(self):
        self.id_index_dict = {}
        for index, skeleton in enumerate(self.skeletons):
            self.id_index_dict.setdefault(skeleton.id, index)

    @property
    def shape(self):
        return self.skeletons.shape

    @property
    def min(self):
        skeleton_min = [skeleton.nodes[["z", "y", "x"]].min().to_numpy() for skeleton in self.skeletons]
        if not skeleton_min: raise ValueError("NeuronList contains no valid nodes.")
        return np.nanmin(np.vstack(skeleton_min), axis=0)

    @property
    def max(self):
        skeleton_max = [skeleton.nodes[["z", "y", "x"]].max().to_numpy() for skeleton in self.skeletons]
        if not skeleton_max: raise ValueError("NeuronList contains no valid nodes.")
        return np.nanmax(np.vstack(skeleton_max), axis=0)

    @staticmethod
    def fix_skeletons_root(skeletons: navis.NeuronList | list[navis.TreeNeuron]):
        fixed_skeletons = []
        for i, skeleton in enumerate(skeletons):
            nodes = skeleton.nodes.copy()
            self_parent_mask = (nodes["node_id"] == nodes["parent_id"])
            fixed_node_ids = nodes.loc[self_parent_mask, "node_id",].tolist()
            nodes.loc[self_parent_mask, "parent_id"] = -1
            fixed_skeleton = navis.TreeNeuron(nodes,
                                              id=getattr(skeleton, "id", None), name=getattr(skeleton, "name", None),)
            fixed_skeletons.append(fixed_skeleton)
        return navis.NeuronList(fixed_skeletons)

    def scale(self, factor: list[float]) -> navis.NeuronList:
        """Scale skeleton coordinates in [z, y, x] order."""
        scale_factor = np.asarray(factor, dtype=np.float64)

        if scale_factor.shape != (3,):
            raise ValueError(
                "factor must contain exactly three values in "
                "[z_factor, y_factor, x_factor] order."
            )

        if not np.all(np.isfinite(scale_factor)):
            raise ValueError("All scale factors must be finite numbers.")

        if np.any(scale_factor <= 0):
            raise ValueError("All scale factors must be greater than zero.")

        coordinate_columns = ["z", "y", "x"]
        scaled_skeletons = []

        for skeleton_index, skeleton in enumerate(self.skeletons):
            nodes = skeleton.nodes.copy().reset_index(drop=True)

            missing_columns = set(coordinate_columns).difference(nodes.columns)
            if missing_columns:
                raise ValueError(
                    f"Skeleton {skeleton_index} is missing coordinate columns: "
                    f"{sorted(missing_columns)}."
                )

            if not nodes.empty:
                coordinates = nodes[coordinate_columns].apply(
                    pd.to_numeric,
                    errors="coerce",
                ).to_numpy(dtype=np.float64)

                if not np.all(np.isfinite(coordinates)):
                    raise ValueError(
                        f"Skeleton {skeleton_index} contains non-finite coordinates."
                    )

                scaled_coordinates = coordinates * scale_factor[None, :]

                for axis, column in enumerate(coordinate_columns):
                    original_dtype = nodes[column].dtype
                    values = scaled_coordinates[:, axis]

                    if pd.api.types.is_float_dtype(original_dtype):
                        values = values.astype(original_dtype, copy=False)

                    nodes[column] = values

            scaled_skeleton = skeleton.copy()
            scaled_skeleton.nodes = nodes
            scaled_skeletons.append(scaled_skeleton)

        self.skeletons = navis.NeuronList(scaled_skeletons)
        self._build_id_index_dict()

        return self.skeletons

    def increase_radius(self, soma_radius: float = 4.0,
                        dendrite_radius: float = 2.0, axon_radius: float = 1.0,) -> navis.NeuronList:
        """Increase radii according to SWC compartment labels.

        Existing radii larger than the specified target are preserved.

        SWC labels:
            1: soma
            2: axon
            3: dendrite
            4: apical dendrite

        Parameters
        ----------
        soma_radius : float
            Minimum radius of soma nodes.
        dendrite_radius : float
            Minimum radius of dendrite and apical-dendrite nodes.
        axon_radius : float
            Minimum radius of axon nodes.

        Returns
        -------
        navis.NeuronList
            Updated neuron list.
        """
        target_radii = {
            "soma": float(soma_radius),
            "dendrite": float(dendrite_radius),
            "axon": float(axon_radius),
        }

        for name, radius in target_radii.items():
            if not np.isfinite(radius) or radius < 0:
                raise ValueError(
                    f"{name}_radius must be a finite non-negative number, "
                    f"but got {radius!r}."
                )

        updated_skeletons = []

        for skeleton_index, skeleton in enumerate(self.skeletons):
            nodes = skeleton.nodes.copy().reset_index(drop=True)

            required_columns = {"label", "radius"}
            missing_columns = required_columns.difference(nodes.columns)

            if missing_columns:
                raise ValueError(
                    f"Neuron {skeleton_index} is missing columns: "
                    f"{sorted(missing_columns)}."
                )

            if nodes.empty:
                updated_skeletons.append(skeleton.copy())
                continue

            labels = pd.to_numeric(
                nodes["label"],
                errors="coerce",
            ).to_numpy(dtype=np.float64)

            radii = pd.to_numeric(
                nodes["radius"],
                errors="coerce",
            ).to_numpy(dtype=np.float64)

            soma_mask = labels == 1
            axon_mask = labels == 2

            # Both basal/general dendrite and apical dendrite.
            dendrite_mask = np.isin(labels, [3, 4])

            def set_minimum_radius(
                    mask: np.ndarray,
                    target_radius: float,
            ) -> None:
                if not np.any(mask):
                    return

                current_radii = radii[mask]

                # Missing radii are replaced; existing larger radii are retained.
                radii[mask] = np.where(
                    np.isfinite(current_radii),
                    np.maximum(current_radii, target_radius),
                    target_radius,
                )

            set_minimum_radius(soma_mask, soma_radius)
            set_minimum_radius(dendrite_mask, dendrite_radius)
            set_minimum_radius(axon_mask, axon_radius)

            # Preserve the original floating-point precision and avoid
            # Pandas LossySetitemError.
            original_dtype = nodes["radius"].dtype

            if pd.api.types.is_float_dtype(original_dtype):
                radii = radii.astype(original_dtype, copy=False)

            nodes["radius"] = radii

            updated_skeleton = skeleton.copy()
            updated_skeleton.nodes = nodes
            updated_skeletons.append(updated_skeleton)

        self.skeletons = navis.NeuronList(updated_skeletons)

        return self.skeletons

    def plot3d(self, palette: str = "turbo"):
        """Plot every disconnected skeleton component with a different color.

        A disconnected component is defined by SWC topology: nodes belong to
        the same component only when they can be reached through ``parent_id``
        links. Therefore, multiple independent trees stored in the same SWC
        file are rendered with different colors.

        Parameters
        ----------
        palette : str, default="turbo"
            Name of a Matplotlib colormap used to generate component colors.

        Returns
        -------
        object
            The Octarine viewer returned by :func:`navis.plot3d`.
        """
        component_skeletons = []

        for skeleton_index, skeleton in enumerate(self.skeletons):
            nodes = skeleton.nodes

            required_columns = {"node_id", "parent_id"}
            missing_columns = required_columns.difference(nodes.columns)
            if missing_columns:
                raise ValueError(
                    f"Skeleton {skeleton_index} is missing columns: "
                    f"{sorted(missing_columns)}."
                )

            if nodes.empty:
                continue

            node_id_values = pd.to_numeric(
                nodes["node_id"],
                errors="coerce",
            ).to_numpy(dtype=np.float64)

            if (
                    np.any(~np.isfinite(node_id_values))
                    or np.any(node_id_values != np.floor(node_id_values))
            ):
                raise ValueError(
                    f"Skeleton {skeleton_index} contains invalid node IDs."
                )

            node_ids = node_id_values.astype(np.int64)

            if len(np.unique(node_ids)) != len(node_ids):
                duplicated_ids = pd.Series(node_ids)[
                    pd.Series(node_ids).duplicated(keep=False)
                ].unique()
                raise ValueError(
                    f"Skeleton {skeleton_index} contains duplicate node IDs: "
                    f"{duplicated_ids[:20].tolist()}."
                )

            parent_values = pd.to_numeric(
                nodes["parent_id"],
                errors="coerce",
            ).to_numpy(dtype=np.float64)

            # Build an undirected topology graph. Isolated nodes are added
            # explicitly, so each isolated node becomes its own component.
            topology_graph = nx.Graph()
            topology_graph.add_nodes_from(node_ids.tolist())

            valid_parent_mask = (
                    np.isfinite(parent_values)
                    & (parent_values >= 0)
                    & (parent_values == np.floor(parent_values))
            )

            if np.any(valid_parent_mask):
                child_ids = node_ids[valid_parent_mask]
                parent_ids = parent_values[valid_parent_mask].astype(np.int64)
                existing_parent_mask = np.isin(parent_ids, node_ids)

                topology_graph.add_edges_from(
                    zip(
                        child_ids[existing_parent_mask].tolist(),
                        parent_ids[existing_parent_mask].tolist(),
                    )
                )

            connected_components = sorted(
                nx.connected_components(topology_graph),
                key=len,
                reverse=True,
            )

            for component_index, component_node_ids in enumerate(
                    connected_components
            ):
                # navis.subset_neuron preserves the skeleton metadata and also
                # removes connectors whose parent nodes are not retained.
                component = navis.subset_neuron(
                    skeleton,
                    subset=component_node_ids,
                    inplace=False,
                    keep_disc_cn=False,
                )

                # Assign a unique display ID/name to avoid ambiguity in the
                # viewer when several components originate from one SWC.
                original_id = getattr(skeleton, "id", skeleton_index)
                original_name = getattr(
                    skeleton,
                    "name",
                    f"skeleton_{skeleton_index}",
                )
                component.id = f"{original_id}_component_{component_index + 1}"
                component.name = (
                    f"{original_name}_component_{component_index + 1}"
                )
                component_skeletons.append(component)

        if not component_skeletons:
            raise ValueError("There are no non-empty skeletons to plot.")

        number_of_components = len(component_skeletons)

        colormap = plt.get_cmap(palette)

        # Uniformly sample colors from the colormap
        color_positions = np.linspace(
            0.0,
            1.0,
            number_of_components,
        )

        # Randomly assign sampled colors to skeleton components
        np.random.shuffle(color_positions)

        component_colors = [
            colormap(position)
            for position in color_positions
        ]

        viewer = navis.plot3d(
            navis.NeuronList(component_skeletons),
            color=component_colors,
            backend="octarine",
        )

        # White background
        viewer.set_bgcolor("white")

        viewer.show(start_loop=True)

        return viewer

    def get_skeletons_index(self, new_skeletons: navis.NeuronList) -> list[int]:
        result_index = []
        for new_skeleton in new_skeletons:
            index = self.id_index_dict.get(new_skeleton.id)
            result_index.append(index)
        assert(len(result_index) == len(new_skeletons))
        return result_index

    def get_nearest_skeleton_node(
            self,
            coordinate: Iterable[float],
            skeleton_index: int,
            return_distance: bool = False,
    ) -> pd.Series | tuple[pd.Series, float]:
        """Return the skeleton node nearest to a coordinate.

        The target skeleton is selected by its zero-based position in
        ``self.skeletons``. This method does not use ``skeleton.id``.

        Parameters
        ----------
        coordinate : Iterable[float]
            Query coordinate in ``[z, y, x]`` order.

        skeleton_index : int
            Zero-based index of the skeleton in ``self.skeletons``.

        return_distance : bool, default=False
            If ``True``, also return the Euclidean distance from ``coordinate``
            to the nearest skeleton node.

        Returns
        -------
        pandas.Series or tuple[pandas.Series, float]
            A copy of the nearest row in ``skeleton.nodes``. When
            ``return_distance=True``, the second value is the Euclidean
            distance in the same coordinate unit as the skeleton.

        Raises
        ------
        TypeError
            If ``skeleton_index`` is not an integer.

        IndexError
            If ``skeleton_index`` is outside ``[0, len(self.skeletons))``.

        ValueError
            If the coordinate is invalid, the selected skeleton has no nodes,
            or its coordinate columns contain invalid values.
        """
        if isinstance(skeleton_index, (bool, np.bool_)) or not isinstance(
                skeleton_index,
                (int, np.integer),
        ):
            raise TypeError(
                "skeleton_index must be an integer, "
                f"but got {type(skeleton_index).__name__}."
            )

        skeleton_index = int(skeleton_index)
        skeleton_count = len(self.skeletons)
        if skeleton_index < 0 or skeleton_index >= skeleton_count:
            raise IndexError(
                "skeleton_index is outside the valid range: "
                f"index={skeleton_index}, skeleton_count={skeleton_count}."
            )

        coordinate_zyx = np.asarray(coordinate, dtype=np.float64)
        if coordinate_zyx.shape != (3,):
            raise ValueError(
                "coordinate must contain exactly three values in [z, y, x] order."
            )
        if not np.all(np.isfinite(coordinate_zyx)):
            raise ValueError(
                "coordinate must contain three finite numeric values."
            )

        skeleton = self.skeletons[skeleton_index]
        nodes = skeleton.nodes
        if not isinstance(nodes, pd.DataFrame):
            raise TypeError(
                f"Skeleton at index {skeleton_index} does not contain a valid "
                "nodes DataFrame."
            )
        if nodes.empty:
            raise ValueError(
                f"Skeleton at index {skeleton_index} contains no nodes."
            )

        coordinate_columns = ["z", "y", "x"]
        missing_columns = set(coordinate_columns).difference(nodes.columns)
        if missing_columns:
            raise ValueError(
                f"Skeleton at index {skeleton_index} is missing coordinate "
                f"columns: {sorted(missing_columns)}."
            )

        node_coordinates = nodes[coordinate_columns].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(node_coordinates)):
            raise ValueError(
                f"Skeleton at index {skeleton_index} contains non-finite "
                "node coordinates."
            )

        coordinate_difference = node_coordinates - coordinate_zyx[None, :]
        squared_distances = np.einsum(
            "ij,ij->i",
            coordinate_difference,
            coordinate_difference,
        )
        nearest_position = int(np.argmin(squared_distances))
        nearest_node = nodes.iloc[nearest_position].copy()

        if return_distance:
            nearest_distance = float(
                np.sqrt(squared_distances[nearest_position])
            )
            return nearest_node, nearest_distance

        return nearest_node

    def increase_nodes_density(self, max_distance: float | Iterable[float]) -> navis.NeuronList:
        """Increase the node density of all skeletons by linear interpolation.

        Existing nodes and their node IDs are preserved. For every parent-child
        edge that is too long, intermediate nodes are inserted along the
        straight line between the two original nodes.

        Parameters
        ----------
        max_distance : float or Iterable[float]
            If a scalar is given, it is interpreted as the maximum Euclidean
            distance between adjacent nodes.

            If three values are given, they must be ``[z_max, y_max, x_max]``.
            In that case, the absolute displacement between adjacent nodes in
            each axis will not exceed the corresponding value.

        Returns
        -------
        navis.NeuronList
            The densified neuron list. ``self.skeletons`` is updated in place
            and the updated list is also returned.
        """
        try:
            distance_array = np.asarray(max_distance, dtype=np.float64)
        except (TypeError, ValueError):
            # ``np.asarray`` does not consume generators in some NumPy
            # versions, so materialize a general iterable as a fallback.
            try:
                distance_array = np.asarray(list(max_distance), dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "max_distance must be a positive scalar or an iterable "
                    "containing three positive values."
                ) from error

        if distance_array.ndim == 0:
            distance_mode = "euclidean"
            distance_limit = float(distance_array)
            if not np.isfinite(distance_limit) or distance_limit <= 0:
                raise ValueError("max_distance must be a finite positive number.")
        else:
            distance_mode = "per_axis"
            distance_limit = distance_array.reshape(-1)
            if distance_limit.size != 3:
                raise ValueError(
                    "max_distance must be either a scalar or three values "
                    "in [z_max, y_max, x_max] order."
                )
            if (not np.all(np.isfinite(distance_limit))
                    or np.any(distance_limit <= 0)):
                raise ValueError(
                    "All values in max_distance must be finite and positive."
                )

        coordinate_columns = ["z", "y", "x"]
        required_columns = {
            "node_id", "parent_id", *coordinate_columns,
        }
        dense_skeletons = []

        for skeleton_index, skeleton in enumerate(self.skeletons):
            nodes = skeleton.nodes.copy().reset_index(drop=True)

            missing_columns = required_columns.difference(nodes.columns)
            if missing_columns:
                raise ValueError(
                    f"Neuron {skeleton_index} is missing required node columns: "
                    f"{sorted(missing_columns)}."
                )

            if nodes.empty:
                dense_skeletons.append(skeleton.copy())
                continue

            # SWC node IDs are integer identifiers. Converting them here also
            # avoids lookup failures caused by mixed int/float pandas dtypes.
            node_id_numeric = pd.to_numeric(nodes["node_id"], errors="coerce")
            parent_id_numeric = pd.to_numeric(nodes["parent_id"], errors="coerce")

            invalid_node_id = (
                node_id_numeric.isna()
                | ~np.isfinite(node_id_numeric.to_numpy(dtype=np.float64))
                | (node_id_numeric.to_numpy(dtype=np.float64)
                   != np.floor(node_id_numeric.to_numpy(dtype=np.float64)))
            )
            if np.any(invalid_node_id):
                raise ValueError(
                    f"Neuron {skeleton_index} contains invalid non-integer node IDs."
                )

            node_ids = node_id_numeric.to_numpy(dtype=np.int64)
            if pd.Series(node_ids).duplicated().any():
                duplicate_ids = pd.Series(node_ids)[
                    pd.Series(node_ids).duplicated(keep=False)
                ].unique()
                raise ValueError(
                    f"Neuron {skeleton_index} contains duplicate node IDs: "
                    f"{duplicate_ids[:20].tolist()}."
                )

            parent_values = parent_id_numeric.to_numpy(dtype=np.float64)
            non_root_mask = np.isfinite(parent_values) & (parent_values >= 0)

            if np.any(
                    non_root_mask
                    & (parent_values != np.floor(parent_values))
            ):
                raise ValueError(
                    f"Neuron {skeleton_index} contains non-integer parent IDs."
                )

            coordinates = nodes[coordinate_columns].to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(coordinates)):
                raise ValueError(
                    f"Neuron {skeleton_index} contains non-finite coordinates."
                )

            child_positions = np.flatnonzero(non_root_mask)
            if child_positions.size == 0:
                dense_skeletons.append(skeleton.copy())
                continue

            parent_ids = parent_values[child_positions].astype(np.int64)
            id_to_position = pd.Series(
                np.arange(len(nodes), dtype=np.int64),
                index=node_ids,
            )
            parent_positions_series = id_to_position.reindex(parent_ids)

            if parent_positions_series.isna().any():
                missing_parent_ids = np.unique(
                    parent_ids[parent_positions_series.isna().to_numpy()]
                )
                raise ValueError(
                    f"Neuron {skeleton_index} references missing parent IDs: "
                    f"{missing_parent_ids[:20].tolist()}."
                )

            parent_positions = parent_positions_series.to_numpy(dtype=np.int64)
            edge_vectors = (
                coordinates[child_positions]
                - coordinates[parent_positions]
            )

            if distance_mode == "euclidean":
                edge_distance = np.linalg.norm(edge_vectors, axis=1)
                segment_counts = np.maximum(
                    1,
                    np.ceil(edge_distance / distance_limit).astype(np.int64),
                )
            else:
                axis_ratios = np.abs(edge_vectors) / distance_limit[None, :]
                segment_counts = np.maximum(
                    1,
                    np.ceil(np.max(axis_ratios, axis=1)).astype(np.int64),
                )

            inserted_counts = segment_counts - 1
            edges_to_split = inserted_counts > 0

            if not np.any(edges_to_split):
                dense_skeletons.append(skeleton.copy())
                continue

            split_child_positions = child_positions[edges_to_split]
            split_parent_positions = parent_positions[edges_to_split]
            split_parent_ids = parent_ids[edges_to_split]
            split_segment_counts = segment_counts[edges_to_split]
            split_inserted_counts = inserted_counts[edges_to_split]

            total_new_nodes = int(split_inserted_counts.sum())
            first_new_id = int(node_ids.max()) + 1
            last_new_id = first_new_id + total_new_nodes - 1
            if last_new_id > np.iinfo(np.int64).max:
                raise OverflowError(
                    f"Neuron {skeleton_index} has no available int64 node IDs."
                )

            # Repeat each child row once for every intermediate node that must
            # be inserted on that edge. This keeps arbitrary extra node columns.
            repeated_child_positions = np.repeat(
                split_child_positions,
                split_inserted_counts,
            )
            repeated_parent_positions = np.repeat(
                split_parent_positions,
                split_inserted_counts,
            )
            repeated_segment_counts = np.repeat(
                split_segment_counts,
                split_inserted_counts,
            )

            new_nodes = nodes.iloc[repeated_child_positions].copy().reset_index(
                drop=True
            )
            new_node_ids = np.arange(
                first_new_id,
                first_new_id + total_new_nodes,
                dtype=np.int64,
            )

            # Position within each edge's inserted-node chain: 1, 2, ..., n-1.
            edge_group_starts = np.repeat(
                np.cumsum(split_inserted_counts) - split_inserted_counts,
                split_inserted_counts,
            )
            interpolation_steps = (
                np.arange(total_new_nodes, dtype=np.int64)
                - edge_group_starts
                + 1
            )
            interpolation_fraction = (
                interpolation_steps / repeated_segment_counts
            )

            parent_coordinates = coordinates[repeated_parent_positions]
            child_coordinates = coordinates[repeated_child_positions]
            interpolated_coordinates = (
                parent_coordinates
                + interpolation_fraction[:, None]
                * (child_coordinates - parent_coordinates)
            )

            # ``coordinates`` and the interpolation fractions are float64,
            # while navis/SWC coordinate columns are commonly float32.
            # Pandas 3.x rejects assigning float64 arrays into float32 columns
            # through ``.loc`` because that would be an implicit lossy cast.
            # Assign each complete column explicitly using its original dtype.
            for axis, column in enumerate(coordinate_columns):
                original_dtype = nodes[column].dtype
                values = interpolated_coordinates[:, axis]
                if pd.api.types.is_float_dtype(original_dtype):
                    values = values.astype(original_dtype, copy=False)
                # Integer coordinate columns must be promoted because inserted
                # coordinates can contain fractional values.
                new_nodes[column] = values

            # Full-column assignment avoids pandas' strict in-place dtype rule.
            new_nodes["node_id"] = new_node_ids

            # The first inserted node points to the original parent. Every
            # subsequent inserted node points to the preceding new node.
            new_parent_ids = new_node_ids - 1
            first_in_chain_mask = interpolation_steps == 1
            repeated_original_parent_ids = np.repeat(
                split_parent_ids,
                split_inserted_counts,
            )
            new_parent_ids[first_in_chain_mask] = (
                repeated_original_parent_ids[first_in_chain_mask]
            )
            new_nodes["parent_id"] = new_parent_ids

            # Make every original child point to the last new node in its chain.
            last_new_node_offsets = np.cumsum(split_inserted_counts) - 1
            last_new_node_ids = new_node_ids[last_new_node_offsets]

            # Update through a NumPy array instead of ``.loc`` so pandas does
            # not reject a lossless int64 -> int32 assignment merely because
            # the source array has a wider dtype.
            parent_id_dtype = nodes["parent_id"].dtype
            updated_parent_ids = nodes["parent_id"].to_numpy(copy=True)
            if pd.api.types.is_integer_dtype(parent_id_dtype):
                dtype_info = np.iinfo(parent_id_dtype)
                if (last_new_node_ids.min() < dtype_info.min
                        or last_new_node_ids.max() > dtype_info.max):
                    raise OverflowError(
                        f"Neuron {skeleton_index} requires node IDs outside "
                        f"the range of parent_id dtype {parent_id_dtype}."
                    )
                replacement_parent_ids = last_new_node_ids.astype(
                    parent_id_dtype, copy=False
                )
            else:
                replacement_parent_ids = last_new_node_ids
            updated_parent_ids[split_child_positions] = replacement_parent_ids
            nodes["parent_id"] = updated_parent_ids

            # Radius is a geometric quantity, so interpolate it when possible.
            if "radius" in nodes.columns:
                parent_radius = pd.to_numeric(
                    nodes.iloc[repeated_parent_positions]["radius"],
                    errors="coerce",
                ).to_numpy(dtype=np.float64)
                child_radius = pd.to_numeric(
                    nodes.iloc[repeated_child_positions]["radius"],
                    errors="coerce",
                ).to_numpy(dtype=np.float64)
                valid_radius = np.isfinite(parent_radius) & np.isfinite(child_radius)
                if np.any(valid_radius):
                    interpolated_radius = (
                        parent_radius[valid_radius]
                        + interpolation_fraction[valid_radius]
                        * (child_radius[valid_radius] - parent_radius[valid_radius])
                    )

                    # Apply the same explicit dtype handling to radius. Using
                    # ``.loc`` here can trigger the same LossySetitemError when
                    # the original radius column is float32.
                    radius_values = pd.to_numeric(
                        new_nodes["radius"], errors="coerce"
                    ).to_numpy(dtype=np.float64)
                    radius_values[valid_radius] = interpolated_radius
                    radius_dtype = nodes["radius"].dtype
                    if pd.api.types.is_float_dtype(radius_dtype):
                        radius_values = radius_values.astype(
                            radius_dtype, copy=False
                        )
                    new_nodes["radius"] = radius_values

            # navis commonly stores root/branch/end/slab classification in
            # ``type``. Newly inserted nodes are slab nodes. If ``type`` is a
            # user-defined numerical SWC field, it is left unchanged instead.
            if "type" in nodes.columns:
                observed_types = set(
                    nodes["type"].dropna().astype(str).str.lower().unique()
                )
                navis_node_types = {"root", "branch", "end", "slab"}
                if observed_types and observed_types.issubset(navis_node_types):
                    # Convert categorical columns before assigning a category
                    # that may not already be present.
                    new_nodes["type"] = new_nodes["type"].astype(object)
                    new_nodes.loc[:, "type"] = "slab"

            dense_nodes = pd.concat(
                [nodes, new_nodes],
                axis=0,
                ignore_index=True,
            )
            dense_nodes = dense_nodes.reindex(columns=nodes.columns)

            # Preserve integer ID dtypes where possible.
            for column in ("node_id", "parent_id"):
                original_dtype = skeleton.nodes[column].dtype
                if pd.api.types.is_integer_dtype(original_dtype):
                    dense_nodes[column] = dense_nodes[column].astype(original_dtype)

            # Copying the original neuron preserves its metadata, units,
            # connectors and other attributes. Original node IDs are unchanged,
            # so node-linked metadata remains valid.
            dense_skeleton = skeleton.copy()
            dense_skeleton.nodes = dense_nodes
            dense_skeletons.append(dense_skeleton)
        self.skeletons = navis.NeuronList(dense_skeletons)
        return self.skeletons

    def crop_skeletons(self, boundary: Iterable[Iterable[float]], ) -> navis.NeuronList:
        """Keep skeleton nodes located inside a 3D bounding box.

        This method only filters nodes by coordinates. It does not calculate
        intersections between skeleton segments and the bounding box.

        If a retained node's parent is outside the bounding box, the retained
        node becomes a new root with ``parent_id = -1``.

        Parameters
        ----------
        boundary : Iterable[Iterable[float]]
            Boundary in the form:
            ``[[zmin, zmax], [ymin, ymax], [xmin, xmax]]``.

        Returns
        -------
        navis.NeuronList
            Cropped skeletons. Neurons without any retained nodes are omitted.
        """
        try:
            bounds = np.asarray(boundary, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "boundary must have the form "
                "[[zmin, zmax], [ymin, ymax], [xmin, xmax]]."
            ) from error

        if bounds.shape != (3, 2):
            raise ValueError(
                "boundary must have shape (3, 2): "
                "[[zmin, zmax], [ymin, ymax], [xmin, xmax]]."
            )

        if not np.all(np.isfinite(bounds)):
            raise ValueError("All boundary values must be finite.")

        lower = bounds[:, 0]
        upper = bounds[:, 1]

        if np.any(lower > upper):
            raise ValueError(
                "Each minimum boundary value must be less than or equal "
                "to its corresponding maximum value."
            )

        coordinate_columns = ["z", "y", "x"]
        required_columns = {
            "node_id",
            "parent_id",
            *coordinate_columns,
        }

        cropped_skeletons = []

        for skeleton_index, skeleton in enumerate(self.skeletons):
            nodes = skeleton.nodes.copy().reset_index(drop=True)

            missing_columns = required_columns.difference(nodes.columns)
            if missing_columns:
                raise ValueError(
                    f"Neuron {skeleton_index} is missing columns: "
                    f"{sorted(missing_columns)}."
                )

            if nodes.empty:
                continue

            coordinates = nodes[
                coordinate_columns
            ].to_numpy(dtype=np.float64)

            if not np.all(np.isfinite(coordinates)):
                raise ValueError(
                    f"Neuron {skeleton_index} contains non-finite coordinates."
                )

            # Boundary surfaces are included.
            inside_mask = np.all(
                (coordinates >= lower[None, :])
                & (coordinates <= upper[None, :]),
                axis=1,
            )

            if not np.any(inside_mask):
                continue

            cropped_nodes = (
                nodes.loc[inside_mask]
                .copy()
                .reset_index(drop=True)
            )

            node_ids = pd.to_numeric(
                cropped_nodes["node_id"],
                errors="raise",
            ).to_numpy(dtype=np.int64)

            parent_values = pd.to_numeric(
                cropped_nodes["parent_id"],
                errors="coerce",
            ).to_numpy(dtype=np.float64)

            retained_id_set = set(node_ids.tolist())

            # Nodes whose parents were removed become new roots.
            new_parent_ids = np.full(
                len(cropped_nodes),
                -1,
                dtype=np.int64,
            )

            valid_parent_mask = (
                    np.isfinite(parent_values)
                    & (parent_values >= 0)
                    & (parent_values == np.floor(parent_values))
            )

            if np.any(valid_parent_mask):
                parent_ids = np.zeros(
                    len(cropped_nodes),
                    dtype=np.int64,
                )

                parent_ids[valid_parent_mask] = parent_values[
                    valid_parent_mask
                ].astype(np.int64)

                parent_retained_mask = (
                        valid_parent_mask
                        & np.isin(
                    parent_ids,
                    node_ids,
                )
                )

                new_parent_ids[parent_retained_mask] = parent_ids[
                    parent_retained_mask
                ]

            # Whole-column assignment avoids Pandas LossySetitemError.
            cropped_nodes["node_id"] = node_ids
            cropped_nodes["parent_id"] = new_parent_ids

            # Recalculate Navis node types after cropping.
            if "type" in cropped_nodes.columns:
                child_counts = pd.Series(
                    new_parent_ids[new_parent_ids >= 0]
                ).value_counts()

                node_types = np.full(
                    len(cropped_nodes),
                    "slab",
                    dtype=object,
                )

                root_mask = new_parent_ids < 0
                node_types[root_mask] = "root"

                for position, node_id in enumerate(node_ids):
                    if root_mask[position]:
                        continue

                    number_of_children = int(
                        child_counts.get(int(node_id), 0)
                    )

                    if number_of_children == 0:
                        node_types[position] = "end"
                    elif number_of_children > 1:
                        node_types[position] = "branch"

                cropped_nodes["type"] = node_types

            cropped_skeleton = skeleton.copy()
            cropped_skeleton.nodes = cropped_nodes

            # Retain connectors attached to retained nodes.
            try:
                connectors = cropped_skeleton.connectors

                if (
                        isinstance(connectors, pd.DataFrame)
                        and not connectors.empty
                        and "node_id" in connectors.columns
                ):
                    cropped_skeleton.connectors = connectors.loc[
                        connectors["node_id"].isin(retained_id_set)
                    ].copy()

            except (AttributeError, TypeError, ValueError):
                pass

            cropped_skeletons.append(cropped_skeleton)

        return navis.NeuronList(cropped_skeletons)

    def create_cropped_skeleton_mask(self, cropped_skeletons: navis.NeuronList,
                                     boundary: Iterable[Iterable[int]],) -> np.ndarray:
        """Rasterize radius-aware skeletons into a local ``(Z, Y, X)`` mask.

        ``boundary`` uses half-open intervals::

            [[zmin, zmax), [ymin, ymax), [xmin, xmax)]

        In Python data form, pass it as::

            [[zmin, zmax], [ymin, ymax], [xmin, xmax]]

        The lower bounds are included and the upper bounds are excluded. The
        output shape is therefore ``(zmax-zmin, ymax-ymin, xmax-xmin)``.

        Background voxels are zero. A cropped skeleton whose index in
        ``self.skeletons`` is ``i`` is written with label ``i + 1``.
        Nodes are rasterized as spheres. Parent-child edges are rasterized as
        variable-radius tubes by densely sampling their centerline and
        linearly interpolating the radius.

        Parameters
        ----------
        cropped_skeletons : navis.NeuronList
            Skeletons to rasterize. Their IDs must exist in
            ``self.skeletons``.
        boundary : Iterable[Iterable[int]]
            Integer voxel boundary in
            ``[[zmin, zmax], [ymin, ymax], [xmin, xmax]]`` order. Each axis
            follows ``[minimum, maximum)`` semantics.

        Returns
        -------
        numpy.ndarray
            Label mask in ``(Z, Y, X)`` order.
        """
        if not isinstance(cropped_skeletons, navis.NeuronList):
            try:
                cropped_skeletons = navis.NeuronList(cropped_skeletons)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "cropped_skeletons must be a navis.NeuronList or an "
                    "iterable of navis.TreeNeuron objects."
                ) from error

        try:
            bounds_float = np.asarray(boundary, dtype=np.float64)
        except (TypeError, ValueError):
            try:
                bounds_float = np.asarray(
                    [list(axis_range) for axis_range in boundary],
                    dtype=np.float64,
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "boundary must have the form "
                    "[[zmin, zmax], [ymin, ymax], [xmin, xmax]]."
                ) from error

        if bounds_float.shape != (3, 2):
            raise ValueError(
                "boundary must have shape (3, 2) in "
                "[[zmin, zmax], [ymin, ymax], [xmin, xmax]] order."
            )
        if not np.all(np.isfinite(bounds_float)):
            raise ValueError("All boundary values must be finite numbers.")
        if np.any(bounds_float != np.floor(bounds_float)):
            raise ValueError(
                "boundary values must be integer voxel coordinates."
            )

        bounds = bounds_float.astype(np.int64)
        lower = bounds[:, 0]
        upper = bounds[:, 1]

        # Half-open intervals must contain at least one voxel on every axis.
        if np.any(lower >= upper):
            raise ValueError(
                "For [minimum, maximum) boundaries, every minimum must be "
                "strictly smaller than its corresponding maximum."
            )

        mask_shape_int64 = upper - lower
        if np.any(mask_shape_int64 > np.iinfo(np.intp).max):
            raise OverflowError("The requested mask is too large to allocate.")

        mask_shape = tuple(mask_shape_int64.astype(np.intp).tolist())

        maximum_label = len(self.skeletons)
        if maximum_label <= np.iinfo(np.uint8).max:
            mask_dtype = np.uint8
        elif maximum_label <= np.iinfo(np.uint16).max:
            mask_dtype = np.uint16
        elif maximum_label <= np.iinfo(np.uint32).max:
            mask_dtype = np.uint32
        else:
            mask_dtype = np.uint64

        mask = np.zeros(mask_shape, dtype=mask_dtype)

        coordinate_columns = ["z", "y", "x"]
        required_columns = {
            "node_id",
            "parent_id",
            "radius",
            *coordinate_columns,
        }

        # Adjacent tube samples move by at most half a voxel along every axis.
        sample_step = 0.5

        for cropped_index, skeleton in enumerate(cropped_skeletons):
            original_index = self.id_index_dict.get(
                getattr(skeleton, "id", None)
            )
            if original_index is None:
                raise ValueError(
                    f"Cropped skeleton at index {cropped_index} with id "
                    f"{getattr(skeleton, 'id', None)!r} does not exist in "
                    "self.skeletons."
                )

            label_value = original_index + 1
            nodes = skeleton.nodes.copy().reset_index(drop=True)

            missing_columns = required_columns.difference(nodes.columns)
            if missing_columns:
                raise ValueError(
                    f"Cropped skeleton {cropped_index} is missing required "
                    f"columns: {sorted(missing_columns)}."
                )
            if nodes.empty:
                continue

            coordinates = nodes[
                coordinate_columns
            ].to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(coordinates)):
                raise ValueError(
                    f"Cropped skeleton {cropped_index} contains non-finite "
                    "coordinates."
                )

            radii = pd.to_numeric(
                nodes["radius"], errors="coerce"
            ).to_numpy(dtype=np.float64)
            if np.any(~np.isfinite(radii)):
                raise ValueError(
                    f"Cropped skeleton {cropped_index} contains non-finite "
                    "radius values."
                )
            if np.any(radii < 0):
                raise ValueError(
                    f"Cropped skeleton {cropped_index} contains negative "
                    "radius values."
                )

            node_id_values = pd.to_numeric(
                nodes["node_id"], errors="coerce"
            ).to_numpy(dtype=np.float64)
            parent_id_values = pd.to_numeric(
                nodes["parent_id"], errors="coerce"
            ).to_numpy(dtype=np.float64)

            invalid_node_id = (
                ~np.isfinite(node_id_values)
                | (node_id_values != np.floor(node_id_values))
            )
            if np.any(invalid_node_id):
                raise ValueError(
                    f"Cropped skeleton {cropped_index} contains invalid "
                    "node IDs."
                )

            node_ids = node_id_values.astype(np.int64)
            duplicated_mask = pd.Series(node_ids).duplicated(
                keep=False
            ).to_numpy()
            if np.any(duplicated_mask):
                duplicate_ids = np.unique(node_ids[duplicated_mask])
                raise ValueError(
                    f"Cropped skeleton {cropped_index} contains duplicate "
                    f"node IDs: {duplicate_ids[:20].tolist()}."
                )

            id_to_position = {
                int(node_id): position
                for position, node_id in enumerate(node_ids)
            }

            def write_nearest_voxel(center_zyx: np.ndarray) -> None:
                """Write the nearest valid voxel using [lower, upper)."""
                voxel = np.rint(center_zyx).astype(np.int64)
                if np.all((voxel >= lower) & (voxel < upper)):
                    local = voxel - lower
                    mask[local[0], local[1], local[2]] = label_value

            def write_ball(
                    center_zyx: np.ndarray,
                    radius: float,
            ) -> None:
                """Write a sphere clipped to the half-open mask boundary."""
                center_zyx = np.asarray(center_zyx, dtype=np.float64)
                radius = float(radius)

                # A zero-radius sample still occupies its nearest voxel when
                # that voxel lies inside [lower, upper).
                if radius <= 0:
                    write_nearest_voxel(center_zyx)
                    return

                # candidate_upper is an inclusive voxel index. Since boundary
                # uses [lower, upper), the last valid index is upper - 1.
                candidate_lower = np.maximum(
                    np.ceil(center_zyx - radius).astype(np.int64),
                    lower,
                )
                candidate_upper = np.minimum(
                    np.floor(center_zyx + radius).astype(np.int64),
                    upper - 1,
                )

                if np.any(candidate_lower > candidate_upper):
                    return

                z_values = np.arange(
                    candidate_lower[0], candidate_upper[0] + 1,
                    dtype=np.int64,
                )
                y_values = np.arange(
                    candidate_lower[1], candidate_upper[1] + 1,
                    dtype=np.int64,
                )
                x_values = np.arange(
                    candidate_lower[2], candidate_upper[2] + 1,
                    dtype=np.int64,
                )

                zz, yy, xx = np.meshgrid(
                    z_values,
                    y_values,
                    x_values,
                    indexing="ij",
                )

                squared_distance = (
                    (zz - center_zyx[0]) ** 2
                    + (yy - center_zyx[1]) ** 2
                    + (xx - center_zyx[2]) ** 2
                )
                inside_ball = squared_distance <= radius ** 2

                if not np.any(inside_ball):
                    # A very small sphere around a non-integer center may not
                    # contain an integer voxel center. Keep the centerline.
                    write_nearest_voxel(center_zyx)
                    return

                global_z = zz[inside_ball]
                global_y = yy[inside_ball]
                global_x = xx[inside_ball]

                mask[
                    global_z - lower[0],
                    global_y - lower[1],
                    global_x - lower[2],
                ] = label_value

            # Draw nodes, including roots and isolated nodes.
            for coordinate, radius in zip(coordinates, radii):
                write_ball(coordinate, radius)

            # Draw each valid parent-child edge as a variable-radius tube.
            for child_position, parent_value in enumerate(parent_id_values):
                if (
                    not np.isfinite(parent_value)
                    or parent_value < 0
                    or parent_value != np.floor(parent_value)
                ):
                    continue

                parent_position = id_to_position.get(int(parent_value))
                if parent_position is None:
                    continue

                parent_coordinate = coordinates[parent_position]
                child_coordinate = coordinates[child_position]
                parent_radius = radii[parent_position]
                child_radius = radii[child_position]

                displacement = child_coordinate - parent_coordinate
                maximum_axis_distance = float(
                    np.max(np.abs(displacement))
                )
                number_of_steps = max(
                    1,
                    int(np.ceil(maximum_axis_distance / sample_step)),
                )

                interpolation_fractions = np.linspace(
                    0.0,
                    1.0,
                    number_of_steps + 1,
                    dtype=np.float64,
                )

                segment_centers = (
                    parent_coordinate[None, :]
                    + interpolation_fractions[:, None]
                    * displacement[None, :]
                )
                segment_radii = (
                    parent_radius
                    + interpolation_fractions
                    * (child_radius - parent_radius)
                )

                # Endpoints were already drawn as nodes.
                for center, radius in zip(
                    segment_centers[1:-1],
                    segment_radii[1:-1],
                ):
                    write_ball(center, radius)

        return mask

    def get_path_to_soma_or_root(
            self,
            skeleton_index: int,
            coordinate: Iterable[float],
            nearest_node_position: int | pd.Series | Iterable[float],
            soma_label: int = 1,
    ) -> navis.TreeNeuron:
        """Create the path from a coordinate to the soma or component root.

        The source skeleton is selected using its zero-based index in
        ``self.skeletons``. A new terminal node is created at ``coordinate``
        and connected to the supplied nearest skeleton node. The function then
        follows ``parent_id`` from that node toward the root and stops at the
        first ancestor whose SWC ``label`` equals ``soma_label``. If no such
        node exists on the ancestor chain, the path ends at the root of that
        connected component.

        The returned node table is ordered as::

            soma/root -> ... -> nearest node -> coordinate node

        Following ``parent_id`` from the newly created coordinate node therefore
        gives::

            coordinate -> nearest node -> ... -> soma/root

        Parameters
        ----------
        skeleton_index : int
            Zero-based index of the source skeleton in ``self.skeletons``.

        coordinate : Iterable[float]
            Coordinate of the new terminal node in ``[z, y, x]`` order.

        nearest_node_position : int, pandas.Series, or Iterable[float]
            Nearest node specification in one of three forms:

            - integer: zero-based row position in the selected node table;
            - ``pandas.Series``: a node row containing ``node_id``;
            - three values: a node coordinate in ``[z, y, x]`` order. The
              closest node in the selected skeleton is used.

        soma_label : int, default=1
            SWC compartment label used to identify soma nodes.

        Returns
        -------
        navis.TreeNeuron
            A new skeleton containing the coordinate node and the unique path
            to the first soma ancestor, or to the component root if no soma is
            present on that ancestor chain.

        Raises
        ------
        TypeError
            If ``skeleton_index`` or ``nearest_node_position`` has an unsupported
            type, or the selected skeleton does not contain a nodes DataFrame.

        IndexError
            If ``skeleton_index`` is outside ``[0, len(self.skeletons))``.

        ValueError
            If coordinates, node IDs, parent IDs, or topology are invalid.
        """
        if isinstance(skeleton_index, (bool, np.bool_)) or not isinstance(
                skeleton_index,
                (int, np.integer),
        ):
            raise TypeError(
                "skeleton_index must be an integer, "
                f"but got {type(skeleton_index).__name__}."
            )

        skeleton_index = int(skeleton_index)
        skeleton_count = len(self.skeletons)
        if skeleton_index < 0 or skeleton_index >= skeleton_count:
            raise IndexError(
                "skeleton_index is outside the valid range: "
                f"index={skeleton_index}, skeleton_count={skeleton_count}."
            )

        skeleton = self.skeletons[skeleton_index]
        if not hasattr(skeleton, "nodes") or not isinstance(
                skeleton.nodes,
                pd.DataFrame,
        ):
            raise TypeError(
                f"Skeleton at index {skeleton_index} does not contain a valid "
                "nodes DataFrame."
            )

        coordinate_zyx = np.asarray(coordinate, dtype=np.float64)
        if coordinate_zyx.shape != (3,):
            raise ValueError(
                "coordinate must contain exactly three values in [z, y, x] order."
            )
        if not np.all(np.isfinite(coordinate_zyx)):
            raise ValueError("coordinate must contain three finite values.")

        if isinstance(soma_label, (bool, np.bool_)):
            raise ValueError(
                f"soma_label must be an integer, but got {soma_label!r}."
            )
        try:
            soma_label_as_float = float(soma_label)
            soma_label_value = int(soma_label)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"soma_label must be an integer, but got {soma_label!r}."
            ) from error
        if (
                not np.isfinite(soma_label_as_float)
                or soma_label_as_float != soma_label_value
        ):
            raise ValueError(
                f"soma_label must be an integer, but got {soma_label!r}."
            )

        nodes = skeleton.nodes.copy().reset_index(drop=True)
        if nodes.empty:
            raise ValueError(
                f"Skeleton at index {skeleton_index} contains no nodes."
            )

        required_columns = {"node_id", "parent_id", "z", "y", "x"}
        missing_columns = required_columns.difference(nodes.columns)
        if missing_columns:
            raise ValueError(
                f"Skeleton at index {skeleton_index} is missing required "
                f"columns: {sorted(missing_columns)}."
            )

        node_id_values = pd.to_numeric(
            nodes["node_id"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(node_id_values)) or np.any(
                node_id_values != np.floor(node_id_values)
        ):
            raise ValueError("All node IDs must be finite integers.")
        node_ids = node_id_values.astype(np.int64)

        if len(np.unique(node_ids)) != len(node_ids):
            duplicated_ids = pd.Series(node_ids)[
                pd.Series(node_ids).duplicated(keep=False)
            ].unique()
            raise ValueError(
                "The selected skeleton contains duplicate node IDs: "
                f"{duplicated_ids[:20].tolist()}."
            )

        parent_values = pd.to_numeric(
            nodes["parent_id"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        finite_parent_mask = np.isfinite(parent_values)
        if np.any(
                finite_parent_mask
                & (parent_values != np.floor(parent_values))
        ):
            raise ValueError("All finite parent IDs must be integers.")

        node_coordinates = nodes[["z", "y", "x"]].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(node_coordinates)):
            raise ValueError(
                "The selected skeleton contains non-finite node coordinates."
            )

        id_to_position = {
            int(node_id): position
            for position, node_id in enumerate(node_ids)
        }

        if isinstance(nearest_node_position, pd.Series):
            if "node_id" not in nearest_node_position.index:
                raise ValueError(
                    "nearest_node_position Series must contain a 'node_id' field."
                )
            nearest_id_value = pd.to_numeric(
                pd.Series([nearest_node_position["node_id"]]),
                errors="coerce",
            ).iloc[0]
            if (
                    not np.isfinite(nearest_id_value)
                    or nearest_id_value != np.floor(nearest_id_value)
            ):
                raise ValueError(
                    "nearest_node_position contains an invalid node_id."
                )
            nearest_node_id = int(nearest_id_value)
            if nearest_node_id not in id_to_position:
                raise ValueError(
                    f"Nearest node ID {nearest_node_id} does not exist in "
                    f"skeleton index {skeleton_index}."
                )
            nearest_position = id_to_position[nearest_node_id]

        elif isinstance(nearest_node_position, (bool, np.bool_)):
            raise TypeError(
                "nearest_node_position must not be a boolean value."
            )

        elif isinstance(nearest_node_position, (int, np.integer)):
            nearest_position = int(nearest_node_position)
            if nearest_position < 0 or nearest_position >= len(nodes):
                raise ValueError(
                    "nearest_node_position is outside the node-table range: "
                    f"position={nearest_position}, node_count={len(nodes)}."
                )
            nearest_node_id = int(node_ids[nearest_position])

        else:
            try:
                nearest_coordinate = np.asarray(
                    nearest_node_position,
                    dtype=np.float64,
                )
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "nearest_node_position must be an integer row position, "
                    "a pandas.Series containing node_id, or a [z, y, x] "
                    "coordinate."
                ) from error

            if nearest_coordinate.shape != (3,):
                raise ValueError(
                    "A coordinate-form nearest_node_position must contain "
                    "three values in [z, y, x] order."
                )
            if not np.all(np.isfinite(nearest_coordinate)):
                raise ValueError(
                    "nearest_node_position coordinate must contain finite values."
                )

            difference = node_coordinates - nearest_coordinate[None, :]
            squared_distance = np.einsum(
                "ij,ij->i",
                difference,
                difference,
            )
            nearest_position = int(np.argmin(squared_distance))
            nearest_node_id = int(node_ids[nearest_position])

        if "label" in nodes.columns:
            label_values = pd.to_numeric(
                nodes["label"],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
        else:
            label_values = np.full(len(nodes), np.nan, dtype=np.float64)

        # Trace nearest node -> first soma ancestor, otherwise -> component root.
        path_node_ids: list[int] = []
        current_node_id = nearest_node_id
        visited_node_ids: set[int] = set()

        while True:
            if current_node_id in visited_node_ids:
                raise ValueError(
                    "A cycle was detected while tracing the path to soma/root. "
                    f"Repeated node ID: {current_node_id}."
                )

            visited_node_ids.add(current_node_id)
            path_node_ids.append(current_node_id)
            current_position = id_to_position[current_node_id]

            current_label = label_values[current_position]
            if (
                    np.isfinite(current_label)
                    and current_label == soma_label_value
            ):
                break

            current_parent = parent_values[current_position]
            if not np.isfinite(current_parent) or current_parent < 0:
                break

            parent_node_id = int(current_parent)
            if parent_node_id == current_node_id:
                break
            if parent_node_id not in id_to_position:
                break

            current_node_id = parent_node_id

        endpoint_to_nearest_ids = path_node_ids[::-1]
        path_positions = [
            id_to_position[node_id]
            for node_id in endpoint_to_nearest_ids
        ]
        path_nodes = nodes.iloc[path_positions].copy().reset_index(drop=True)

        path_node_ids_array = np.asarray(
            endpoint_to_nearest_ids,
            dtype=np.int64,
        )
        path_parent_ids = np.full(
            len(path_node_ids_array),
            -1,
            dtype=np.int64,
        )
        if len(path_node_ids_array) > 1:
            path_parent_ids[1:] = path_node_ids_array[:-1]

        path_nodes["node_id"] = path_node_ids_array
        path_nodes["parent_id"] = path_parent_ids

        maximum_node_id = int(np.max(node_ids))
        if maximum_node_id >= np.iinfo(np.int64).max:
            raise OverflowError(
                "Cannot create a coordinate node because node_id has reached "
                "the int64 maximum."
            )
        coordinate_node_id = maximum_node_id + 1

        coordinate_node = nodes.iloc[[nearest_position]].copy().reset_index(
            drop=True
        )
        coordinate_node["node_id"] = np.asarray(
            [coordinate_node_id],
            dtype=np.int64,
        )
        coordinate_node["parent_id"] = np.asarray(
            [nearest_node_id],
            dtype=np.int64,
        )

        for axis, column in enumerate(["z", "y", "x"]):
            original_dtype = nodes[column].dtype
            if pd.api.types.is_float_dtype(original_dtype):
                coordinate_node[column] = np.asarray(
                    [coordinate_zyx[axis]],
                    dtype=original_dtype,
                )
            else:
                coordinate_node[column] = np.asarray(
                    [coordinate_zyx[axis]],
                    dtype=np.float64,
                )

        if "type" in coordinate_node.columns:
            coordinate_node["type"] = "end"

        new_nodes = pd.concat(
            [path_nodes, coordinate_node],
            ignore_index=True,
            sort=False,
        )
        new_nodes["node_id"] = pd.to_numeric(
            new_nodes["node_id"],
            errors="raise",
        ).astype(np.int64)
        new_nodes["parent_id"] = pd.to_numeric(
            new_nodes["parent_id"],
            errors="raise",
        ).astype(np.int64)

        if "type" in new_nodes.columns:
            new_node_ids = new_nodes["node_id"].to_numpy(dtype=np.int64)
            new_parent_ids = new_nodes["parent_id"].to_numpy(dtype=np.int64)
            child_counts = pd.Series(
                new_parent_ids[new_parent_ids >= 0]
            ).value_counts()

            node_types = np.full(len(new_nodes), "slab", dtype=object)
            root_mask = new_parent_ids < 0
            node_types[root_mask] = "root"

            for position, node_id in enumerate(new_node_ids):
                if root_mask[position]:
                    continue
                number_of_children = int(
                    child_counts.get(int(node_id), 0)
                )
                if number_of_children == 0:
                    node_types[position] = "end"
                elif number_of_children > 1:
                    node_types[position] = "branch"

            new_nodes["type"] = node_types

        result_skeleton = skeleton.copy()
        result_skeleton.nodes = new_nodes

        try:
            connectors = skeleton.connectors
            if (
                    isinstance(connectors, pd.DataFrame)
                    and not connectors.empty
                    and "node_id" in connectors.columns
            ):
                retained_ids = set(endpoint_to_nearest_ids)
                result_skeleton.connectors = connectors.loc[
                    connectors["node_id"].isin(retained_ids)
                ].copy()
        except (AttributeError, TypeError, ValueError):
            pass

        return result_skeleton


if __name__ == '__main__':
    path = r"E:\Albert_BigFile\Data\260618_SkNeXt_dataset\Skeleton1"
    Skeleton = SkeletonManager(path)
    viewer = Skeleton.plot3d(palette="Dark2")
    # boudary = np.array([[0, 500],[8000,8200],[8000,8200]])
    # cropped = Skeleton.crop_skeletons(boudary)
    # print(len(cropped))
    # print(type(cropped))
    # print(Skeleton.get_skeletons_index(cropped))
    # print(cropped[0].nodes)
    # mask = Skeleton.create_cropped_skeleton_mask(cropped, boudary)
    # tifffile.imwrite(r"E:\Albert_BigFile\Data\260618_SkNeXt_dataset\mask.tif", mask)
    # viewer = navis.plot3d(cropped, backend="octarine", )
    # viewer.show(start_loop=True)