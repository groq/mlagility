import os
from typing import Dict, List


def _convert_to_dict(label_list: List[str]) -> Dict[str, List[str]]:
    """
    Convert label list into a dictionary of labels
    """
    label_dict = {}
    for item in label_list:
        try:
            label_key, label_value = item.split("::")
        except ValueError:
            # FIXME: Create a proper warning for this once we have the right
            # infrastructure for doing so.
            # https://github.com/groq/mlagility/issues/55
            print(
                f"Warning: Malformed label {item} found. ",
                "Each label must have the format key::value1,value2,... ",
            )
        label_value = label_value.split(",")
        label_dict[label_key] = label_value
    return label_dict


def load_from_file(file_path: str) -> Dict[str, List[str]]:
    """
    This function extracts labels from a Python file.
    Labels must be in the first line of a Python file and start with "# labels: "
    Each label must have the format "key::value1,value2,..."

    Example:
        "# labels: author::google test_group::daily,monthly"
    """
    # Open file
    with open(file_path, encoding="utf-8") as f:
        first_line = f.readline()

    # Return label dict
    if "# labels:" in first_line:
        label_list = first_line.replace("\n", "").split(" ")[2:]
        return _convert_to_dict(label_list)
    else:
        return {}


def load_from_cache(cache_dir: str, build_name: str) -> Dict[str, List[str]]:
    """
    Loads labels from the cache directory
    """
    # Open file
    labels_dir = os.path.join(cache_dir, "labels")
    file_path = os.path.join(labels_dir, f"{build_name}.txt")
    with open(file_path, encoding="utf-8") as f:
        first_line = f.readline()

    # Return label dict
    label_list = first_line.replace("\n", "").split(" ")
    return _convert_to_dict(label_list)


def save_to_cache(cache_dir: str, build_name: str, label_dict: Dict[str, List[str]]):
    """
    Save labels as a stand-alone file as part of the cache directory
    """
    labels_list = [f"{k}::{','.join(label_dict[k])}" for k in label_dict.keys()]

    # Create labels folder if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    labels_dir = os.path.join(cache_dir, "labels")
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    # Save labels to cache
    file_path = os.path.join(labels_dir, f"{build_name}.txt")
    with open(file_path, "w", encoding="utf8") as fp:
        fp.write(" ".join(labels_list))
