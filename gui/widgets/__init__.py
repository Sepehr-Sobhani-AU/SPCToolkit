"""
This module is the centralised import for all custom widgets used in the GUI.
"""
from .pcd_viewer_widget import PCDViewerWidget
from .tree_structure_widget import TreeStructureWidget
from .process_overlay_widget import ProcessOverlayWidget
from .splash_screen import SplashScreen

__all__ = ["PCDViewerWidget", "TreeStructureWidget", "ProcessOverlayWidget", "SplashScreen"]
