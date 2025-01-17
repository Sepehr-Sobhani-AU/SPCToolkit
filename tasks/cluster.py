from core.data_node import DataNode


class Cluster:
    def __init__(self, params: dict):
        self.params = params

    def execute(self, data_node: DataNode):
        print("Cluster task...", data_node.name, data_node.tags, self.params)
