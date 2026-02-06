# TODO: Docstrings

from core.entities.point_cloud import PointCloud
from core.entities.masks import Masks


class ApplyMasks:

    def __init__(self, point_cloud: PointCloud, masks: Masks):
        self.point_cloud = point_cloud
        self.masks = masks.mask

    def execute(self):

        point_cloud_subset = self.point_cloud.get_subset(mask=self.masks)
        return point_cloud_subset
