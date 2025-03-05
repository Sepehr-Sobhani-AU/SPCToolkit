# TODO: Docstrings
# TODO: Validations

from core.point_cloud import PointCloud
from core.values import Values


class ApplyValues:

    def __init__(self, point_cloud: PointCloud, values: Values):
        self.point_cloud = point_cloud
        self.values = values.values

    def execute(self):
        self.point_cloud.add_attribute("values", self.values)
        return self.point_cloud