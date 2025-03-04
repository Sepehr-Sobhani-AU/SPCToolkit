# TODO: Metadata list are hard-coded in this class and will require changes to the code whenever a new metadata type is added. A better approach would be to use a factory pattern to create the metadata instances based on the metadata type. This way, new metadata types can be added without modifying the MetadataManager class. The factory pattern can be implemented as a separate class or as a static method within the Metadata class. The factory method would take the metadata type as an argument and return an instance of the corresponding metadata class. This would decouple the MetadataManager from the specific metadata types and make the code more extensible.
import uuid
from typing import Any, Dict, List


class AnalysisResult:
    """
    Represents the result of an analysis or task performed on data nodes.

    Attributes:
        uuid (UUID): A unique identifier for the analysis result.
        name (str): The params or description of the analysis result.
        params (dict): Parameters used for the analysis.
        data (object): The resulting data from the analysis (e.g., cluster_labels, subsamples).
        depends_on (list[UUID]): The UUIDs of DataNodes that this result depends on.
    """

    def __init__(
        self,
        name: str,
        data: Any,
        params: Dict[str, Any],
        depends_on: List[uuid.UUID]
    ):
        """
        Initializes an AnalysisResult instance.

        Args:
            name (str): The params or description of the analysis result.
            data (Any): The resulting data from the analysis.
            params (dict): Parameters used for the analysis.
            depends_on (List[UUID]): UUIDs of DataNodes that this result depends on.
        """
        self.uuid: uuid.UUID = uuid.uuid4()
        self.name: str = name
        self.params: Dict[str, Any] = params
        self.data: Any = data
        self.depends_on: List[uuid.UUID] = depends_on

    def get_metadata(self) -> Dict[str, Any]:
        """
        Retrieves metadata for the analysis result.

        Returns:
            dict: A dictionary containing metadata such as the params, parameters, and dependencies.
        """
        return {
            "uid": str(self.uuid),
            "params": self.name,
            "params": self.params,
            "depends_on": [str(dep) for dep in self.depends_on]
        }

    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Updates metadata for the analysis result.

        Args:
            metadata (dict): A dictionary containing metadata to update.
        """
        self.name = metadata.get("params", self.name)
        self.params = metadata.get("params", self.params)
        if "depends_on" in metadata:
            self.depends_on = metadata["depends_on"]

    def __repr__(self) -> str:
        """
        Provides a string representation of the AnalysisResult instance.

        Returns:
            str: A string describing the analysis result.
        """
        return f"AnalysisResult(params={self.name}, uid={self.uuid})"
