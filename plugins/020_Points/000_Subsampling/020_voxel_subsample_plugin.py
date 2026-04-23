from typing import Dict, Any, List, Tuple
import numpy as np
import torch

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.masks import Masks
from config.config import global_variables


class VoxelSubsamplePlugin(Plugin):
    """
    GPU-accelerated voxel-grid subsampling.

    Hashes points into integer voxel keys on CUDA and keeps one original
    point per voxel (no centroid synthesis).
    """

    def get_name(self) -> str:
        return "voxel_subsample"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "voxel_size": {
                "type": "float",
                "default": 0.05,
                "min": 0.001,
                "max": 10.0,
                "label": "Voxel Size",
                "description": "Edge length of voxels; one original point is kept per voxel."
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Voxel subsample requires a GPU.")

        point_cloud: PointCloud = data_node.data
        voxel_size = float(params["voxel_size"])
        n = len(point_cloud.points)

        global_variables.global_progress = (
            None, f"Voxel subsample ({n:,} pts, voxel={voxel_size})..."
        )

        device = torch.device('cuda')
        pts = torch.from_numpy(np.asarray(point_cloud.points, dtype=np.float32)).to(device)

        origin = pts.min(dim=0).values
        keys = torch.floor((pts - origin) / voxel_size).to(torch.int64)

        # Shift to non-negative and pack three 21-bit components into one int64.
        keys_shifted = keys - keys.min(dim=0).values
        max_extent = int(keys_shifted.max().item()) + 1
        if max_extent >= (1 << 21):
            raise RuntimeError(
                f"Voxel grid too large to pack (extent={max_extent}); increase voxel_size."
            )
        packed = (
            (keys_shifted[:, 0] << 42)
            | (keys_shifted[:, 1] << 21)
            | keys_shifted[:, 2]
        )

        unique_packed, inverse = torch.unique(packed, return_inverse=True)
        v = unique_packed.numel()

        if global_variables.global_cancel_event.is_set():
            raise RuntimeError("Voxel subsample cancelled.")

        global_variables.global_progress = (60, f"Selecting {v:,} voxel representatives...")

        # One representative per voxel: smallest original index in each voxel.
        point_idx = torch.arange(n, device=device, dtype=torch.int64)
        rep = torch.full((v,), n, dtype=torch.int64, device=device)
        rep.scatter_reduce_(0, inverse, point_idx, reduce="amin", include_self=True)

        chosen = rep.cpu().numpy()
        mask = np.zeros(n, dtype=bool)
        mask[chosen] = True

        global_variables.global_progress = (100, f"Voxel subsample done ({v:,} of {n:,} kept).")
        return Masks(mask), "masks", [data_node.uid]
