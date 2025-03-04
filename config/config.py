# TODO: Docstrings
# TODO: Define types



# Global variables
class GlobalVariables:
    def __init__(self):
        self.global_pcd_viewer_widget = None
        self.global_tree_structure_widget = None
        self.global_data_nodes: core.data_nodes = None
        self.global_data_manager: core.data_manager = None


# Singleton instance
global_variables = GlobalVariables()
