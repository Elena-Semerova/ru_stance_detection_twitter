import yaml
from typing import Any

def read_yml_file(file_path: str) -> Any:
    with open(file_path, "r") as file:
        result = yaml.safe_load(file)

    return result