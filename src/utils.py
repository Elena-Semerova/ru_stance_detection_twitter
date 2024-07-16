import yaml


def read_yml_file(file_path):
    with open(file_path, "r") as file:
        result = yaml.safe_load(file)

    return result