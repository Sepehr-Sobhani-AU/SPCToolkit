# TODO: The list of metadata items are hard-coded in this class and are not extensible. A better approach would be to use a metadata dictionary to store the metadata items and their values. This would allow adding new metadata items without modifying the class. The metadata dictionary could be initialized with default values and updated as needed. The get_metadata and update_metadata methods could be modified to use the metadata dictionary to retrieve and update metadata items. This would make the class more flexible and easier to maintain.

import uuid
from typing import List, Any, Dict


class DataNode:
    """
    Represents a single unit of data, which can either be raw data or derived data.

    Attributes:
        uid (UUID): Unique identifier for the data node.
        params (str): A descriptive params for the data node.
        data (Any): The actual data object stored in this node. Supported types:
            - PointCloud: Primary 3D point cloud data with coordinates, colors, normals, and attributes
            - masks: Boolean arrays for point selection/filtering
            - cluster_labels: Integer arrays assigning points to clusters
            - values: Arbitrary scalar values per point
            - eigenvalues: Eigenvalue-based geometric features
            - colors: RGB color arrays for visualization
            - dist_to_ground: Distance-to-ground measurements
            - feature_classes: Feature classification (Tree, Car, Building, etc.) for ML
            - labels: Cluster labels
            - indexs: Index arrays for point references
        parent_uid (UUID): The UUID of the parent node, if applicable.
        depends_on (List[UUID]): A list of UUIDs that this node depends on.
        tags (List[str]): Tags to classify the node (e.g., 'raw', 'derived', 'subsampled').
    """

    def __init__(
        self,
        params: str = "",
        alias: str = "",
        data: Any = None,
        data_type: str = "",
        data_name: str = "",
        parent_uid: uuid.UUID = None,
        depends_on: List[uuid.UUID] = None,
        tags: List[str] = None
    ):
        """
        Initializes a new DataNode instance.

        Args:
            params (str): A descriptive params for the node.
            data (Any): The data stored in the node.
            parent_uid (UUID, optional): The parent node's UUID, if any. Defaults to None.
            depends_on (List[UUID], optional): List of dependent nodes. Defaults to None.
            tags (List[str], optional): Tags for classification. Defaults to None.
            data_type (str): The type of data stored in the node.
        """
        self.uid: uuid.UUID = uuid.uuid4()
        self.params: str = params
        self.alias: str = alias
        self.data: Any = data
        self.data_type: str = self.data.__class__.__name__ if data_type == "" else data_type
        self.data_name: str = data_name
        self.parent_uid: uuid.UUID = parent_uid
        self.depends_on: List[uuid.UUID] = depends_on if depends_on else []
        self.tags: List[str] = tags if tags else []

    def __repr__(self) -> str:
        """
        Provides a string representation of the DataNode.

        Returns:
            str: String representation including the UUID and params.
        """
        return f"DataNode(uid={self.uid}, params='{self.params}')"
