"""
This module provides a class for managing a collection of DataNode instances,
including methods for adding, retrieving, removing, and validating dependencies between nodes.
"""

import uuid
import logging
from typing import Dict, List, Optional
from core.data_node import DataNode

# Get logger for this module
logger = logging.getLogger(__name__)


class DataNodes:
    """
    Manages a collection of DataNode instances, providing methods for adding, retrieving,
    removing, and validating dependencies between nodes.

    Attributes:
        data_nodes (dict): A dictionary mapping UUIDs to their corresponding DataNode instances.
    """

    def __init__(self):
        """
        Initializes an empty DataNodes manager.
        """
        self.data_nodes: Dict[uuid.UUID, DataNode] = {}

    def add_node(self, node: DataNode) -> uuid.UUID:
        """
        Adds a new DataNode to the collection.

        Args:
            node (DataNode): The DataNode instance to add.

        Returns:
            UUID: The unique identifier of the added DataNode.
        """
        logger.debug(f"DataNodes.add_node() called")
        logger.debug(f"  Node UID: {node.uid}")
        logger.debug(f"  Node type: {node.data_type}")
        logger.debug(f"  Node params: {node.params}")
        logger.debug(f"  Total nodes before: {len(self.data_nodes)}")

        self.data_nodes[node.uid] = node

        logger.debug(f"  Total nodes after: {len(self.data_nodes)}")
        return node.uid

    def remove_node(self, uid: uuid.UUID) -> bool:
        """
        Removes a DataNode instance from the collection.

        Args:
            uid (UUID): UUID of DataNode to remove.

        Returns:
            bool: True if all nodes were removed successfully, False otherwise.
        """

        if uid in self.data_nodes:
            del self.data_nodes[uid]
        else:
            print(f"Warning: DataNode with UUID {uid} not found.")
            return False

        return True

    def get_node(self, uid: uuid.UUID) -> Optional[DataNode]:
        """
        Retrieves a DataNode instance by its UUID.

        Args:
            uid (UUID): The UUID of the DataNode to retrieve.

        Returns:
            Optional[DataNode]: The requested DataNode instance, or None if not found.
        """
        return self.data_nodes.get(uid)

    def update_parent(self, uid: uuid.UUID, new_parent_uid: uuid.UUID) -> bool:
        """
        Updates the parent of a given DataNode, ensuring dependency constraints are met.

        Args:
            uid (uuid.UUID): The unique identifier of the node to update.
            new_parent_uid (uuid.UUID): The unique identifier of the new parent.

        Raises:
            ValueError: If the new parent violates dependency rules.
            KeyError: If the specified DataNode or new parent does not exist.

        Returns:
            bool: True if all nodes were removed successfully, False otherwise.

        """
        if uid not in self.data_nodes:
            print(f"DataNode with UID {uid} not found.")
            return False

        if new_parent_uid not in self.data_nodes:
            print(f"New parent DataNode with UID {new_parent_uid} not found.")
            return False

        # Validate dependencies using the dedicated method
        if not self.validate_dependency(uid):
            print(f"Re-parenting DataNode {uid} to {new_parent_uid} violates dependency rules.")
            return False

        # Update the parent if validation passes
        self.data_nodes[uid].parent_uid = new_parent_uid
        return True

    def validate_dependency(self, uid: uuid.UUID) -> bool:
        """
        Validates whether a list of DataNodes can be safely removed or moved based on dependencies.

        Args:
            uid (UUID): UUID to validate.

        Returns:
            bool: True if there are no dependent DataNodes preventing the operation, False otherwise.
        """
        dependent_nodes = []
        for node in self.data_nodes.values():
            if uid in node.depends_on:
                dependent_nodes.append(node.uid)

        if dependent_nodes:
            print(f"Dependent nodes exist: {dependent_nodes}")
            return False
        return True

    def __repr__(self) -> str:
        """
        Provides a string representation of the DataNodes manager.

        Returns:
            str: A string listing all DataNodes in the collection.
        """
        return f"DataNodes({len(self.data_nodes)} nodes)"

    def list_nodes(self) -> List[uuid.UUID]:
        """
        Lists all DataNode UUIDs in the collection.

        Returns:
            List[str]: A list of DataNode UUIDs.
        """
        return [uid for uid in self.data_nodes.keys()]
