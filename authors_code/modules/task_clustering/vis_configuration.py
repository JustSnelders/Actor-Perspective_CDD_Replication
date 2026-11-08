import yaml
from pathlib import Path


class VisualizationConfiguration:
    def __init__(self, path=Path("modules\\task_clustering\\config_vis.yaml")):
        config = yaml.safe_load(open(path))

        self.node_properties = config["node_properties"]
