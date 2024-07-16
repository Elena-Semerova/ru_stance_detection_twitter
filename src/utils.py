import yaml
from typing import Any

def read_yml_file(file_path: str) -> Any:
    """
    Reading yaml-file

    Params:
    -------
        file_path (str): path for file for reading

    Returns:
    --------
        result (Any): readed file
    """
    with open(file_path, "r") as file:
        result = yaml.safe_load(file)

    return result