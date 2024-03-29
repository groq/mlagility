"""
APIs to help with programmatically parsing the MLAgility report
"""

from typing import Dict, List
import pandas as pd

def get_dict(report_csv: str, columns: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Returns a dictionary where the keys are model names and the values are dictionaries. 
    Each dictionary represents a model with column names as keys and their corresponding values.
    args:
     - report_csv: path to a report.csv file generated by benchit
     - columns: list of column names in the report.csv file whose values will be used to
        populate the dictionary
    """

    # Load the MLAgility report as a dataframe
    dataframe = pd.read_csv(report_csv)

    # Create a nested dictionary with model_name as keys and another dictionary of {column: value} pairs as values
    result = {row[0]: row[1].to_dict() for row in dataframe.set_index("model_name")[columns].iterrows()}

    return result
