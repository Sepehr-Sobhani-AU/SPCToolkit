# TODO: The list of metadata items are hard-coded in this class and are not extensible. A better approach would be to use a metadata dictionary to store the metadata items and their values. This would allow adding new metadata items without modifying the class. The metadata dictionary could be initialized with default values and updated as needed. The get_metadata and update_metadata methods could be modified to use the metadata dictionary to retrieve and update metadata items. This would make the class more flexible and easier to maintain.

import uuid
from typing import List, Any, Dict


class DataNode:
    """
    Represents a single unit of data, which can either be raw data or derived data.

    Attributes:
        uid (UUID): Unique identifier for the data node.
        name (str): A descriptive name for the data node.
        data (Any): The actual data object stored in this node (e.g., point cloud, results).
        parent_uid (UUID): The UUID of the parent node, if applicable.
        depends_on (List[UUID]): A list of UUIDs that this node depends on.
        tags (List[str]): Tags to classify the node (e.g., 'raw', 'derived', 'subsampled').
    """

    def __init__(
        self,
        name: str,
        data: Any,
        parent_uid: uuid.UUID = None,
        depends_on: List[uuid.UUID] = None,
        tags: List[str] = None
    ):
        """
        Initializes a new DataNode instance.

        Args:
            name (str): A descriptive name for the node.
            data (Any): The data stored in the node.
            parent_uid (UUID, optional): The parent node's UUID, if any. Defaults to None.
            depends_on (List[UUID], optional): List of dependent nodes. Defaults to None.
            tags (List[str], optional): Tags for classification. Defaults to None.
        """
        self.uid: uuid.UUID = uuid.uuid4()
        self.name: str = name
        self.data: Any = data
        self.parent_uid: uuid.UUID = parent_uid
        self.depends_on: List[uuid.UUID] = depends_on if depends_on else []
        self.tags: List[str] = tags if tags else []

    def get_metadata(self) -> Dict[str, Any]:
        """
        Retrieves metadata about the DataNode.

        Returns:
            dict: A dictionary containing metadata about the node.
        """
        return {
            "uid": str(self.uid),
            "name": self.name,
            "parent_uid": str(self.parent_uid) if self.parent_uid else None,
            "depends_on": [str(uid) for uid in self.depends_on],
            "tags": self.tags
        }

    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Updates metadata for the DataNode. Only updates provided keys.

        Args:
            metadata (dict): A dictionary containing metadata keys to update.
        """
        if "name" in metadata:
            self.name = metadata["name"]
        if "parent_uid" in metadata:
            self.parent_uid = uuid.UUID(metadata["parent_uid"])
        if "depends_on" in metadata:
            self.depends_on = [uuid.UUID(uid) for uid in metadata["depends_on"]]
        if "tags" in metadata:
            self.tags = metadata["tags"]

    def __repr__(self) -> str:
        """
        Provides a string representation of the DataNode.

        Returns:
            str: String representation including the UUID and name.
        """
        return f"DataNode(uid={self.uid}, name='{self.name}')"
