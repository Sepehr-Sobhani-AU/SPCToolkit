# TODO: Docstrings
# TODO: Define types
import core.data_nodes

# Global variables
class GlobalVariables:
    def __init__(self):
        self.global_pcd_viewer_widget = None
        self.global_tree_structure_widget = None
        self.global_data_nodes: core.data_nodes = None


# Singleton instance
global_variables = GlobalVariables()
