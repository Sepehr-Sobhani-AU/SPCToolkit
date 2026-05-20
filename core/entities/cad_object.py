"""
Backwards-compat shim for the CADObject → VectorFeature rename (2026-05-17).

Old pickled `.pcdtk` projects contain references to
`core.entities.cad_object.CADObject`. This module keeps unpickling working
by aliasing `CADObject` to the renamed `VectorFeature`. New code should
import `VectorFeature` from `core.entities.vector_feature` directly.

Safe to remove once all known project files have been re-saved.
"""

from core.entities.vector_feature import VectorFeature

CADObject = VectorFeature
