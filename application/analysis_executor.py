"""
Analysis Executor — handles asynchronous analysis execution with threading.

Single process at a time. Uses callbacks for progress/completion notification.
Designed for future multi-process extension.
"""
import logging
import time
import threading
from typing import Optional, Type, Dict, Any, Callable

from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.services.reconstruction_service import ReconstructionService
from core.services.cache_service import CacheService
from core.services.analysis_service import AnalysisService

logger = logging.getLogger(__name__)


class AnalysisExecutor:
    """
    Handles asynchronous analysis execution with threading.

    Only one analysis can run at a time (single process, UI blocked).
    Progress and completion are reported via callbacks.
    """

    def __init__(self,
                 reconstruction_service: ReconstructionService,
                 cache_service: CacheService,
                 analysis_service: AnalysisService):
        self._reconstruction_service = reconstruction_service
        self._cache_service = cache_service
        self._analysis_service = analysis_service
        self._is_running = False
        self._thread: Optional[threading.Thread] = None

        # Completion state (read by polling from main thread)
        self._is_completed = False
        self._result_data: Optional[Dict[str, Any]] = None
        self._error_data: Optional[Dict[str, Any]] = None

    def execute(self,
                plugin_class: Type,
                data_node: DataNode,
                params: Dict[str, Any],
                analysis_type: str,
                on_progress: Callable[[int, str], None] = None,
                on_complete: Callable = None,
                on_error: Callable[[str], None] = None) -> None:
        """
        Execute analysis in a background thread.

        Only one analysis can run at a time.

        Args:
            plugin_class: Plugin class to instantiate and execute.
            data_node: Target DataNode.
            params: Analysis parameters.
            analysis_type: Name of the analysis.
            on_progress: Called with (percent, message) during execution.
            on_complete: Called on success — no args, main thread processes result.
            on_error: Called with error message on failure.
        """
        if self._is_running:
            if on_error:
                on_error("Analysis already running")
            return

        self._is_running = True
        self._is_completed = False
        self._result_data = None
        self._error_data = None

        thread = threading.Thread(
            target=self._run_in_thread,
            args=(plugin_class, data_node, params, analysis_type),
            daemon=True
        )
        self._thread = thread
        thread.start()

        logger.info(f"Started analysis thread for '{analysis_type}'")

    def is_running(self) -> bool:
        """Check if an analysis is currently running."""
        return self._is_running

    def check_and_process_completion(self) -> bool:
        """
        Check if analysis completed and return True if done.

        Called periodically from the main thread (via QTimer polling).

        Returns:
            True if analysis completed (success or error), False if still running.
        """
        if not self._is_completed:
            return False

        return True

    def get_result(self) -> Optional[Dict[str, Any]]:
        """
        Get the analysis result after completion.

        Returns:
            Dict with keys: result, result_type, dependencies, data_node,
            analysis_type, params. None if not completed or error.
        """
        return self._result_data

    def get_error(self) -> Optional[str]:
        """
        Get error message if analysis failed.

        Returns:
            Error message string, or None if no error.
        """
        if self._error_data:
            return self._error_data.get('error', 'Unknown error')
        return None

    def cleanup(self):
        """Clean up state after processing completion. Call from main thread."""
        self._is_completed = False
        self._result_data = None
        self._error_data = None
        self._is_running = False
        self._thread = None

    def _run_in_thread(self, plugin_class: Type, data_node: DataNode,
                       params: Dict[str, Any], analysis_type: str):
        """
        Background thread execution.

        Handles reconstruction if needed, executes the plugin,
        and stores results for the main thread to pick up.
        """
        try:
            data_node_to_process = data_node

            # Reconstruct if data is not a point_cloud
            if data_node_to_process.data_type != "point_cloud":
                logger.info("Reconstructing branch in background thread...")
                point_cloud = self._reconstruction_service.reconstruct(
                    str(data_node_to_process.uid)
                )

                # Auto-cache the parent after reconstruction
                if not data_node_to_process.is_cached:
                    self._cache_service.set(str(data_node_to_process.uid), point_cloud)
                    logger.info(
                        f"Auto-cached parent after reconstruction: {data_node_to_process.params}"
                    )

                # Create temporary DataNode with reconstructed PointCloud
                reconstructed_data_node = DataNode(
                    params=data_node_to_process.params,
                    data=point_cloud,
                    data_type="point_cloud"
                )
                reconstructed_data_node.uid = data_node_to_process.uid
                data_node_to_process = reconstructed_data_node

            # Execute analysis via AnalysisService
            logger.info(f"Executing plugin '{analysis_type}' in background thread...")
            result, result_type, dependencies = self._analysis_service.execute(
                plugin_class, data_node_to_process, params
            )

            # Store result for main thread
            self._result_data = {
                'result': result,
                'result_type': result_type,
                'dependencies': dependencies,
                'data_node': data_node,  # Original node (may now be cached)
                'analysis_type': analysis_type,
                'params': params
            }
            self._is_completed = True

        except Exception as e:
            self._error_data = {
                'error': str(e),
                'analysis_type': analysis_type
            }
            self._is_completed = True
            logger.error(f"Exception in analysis thread: {e}")
