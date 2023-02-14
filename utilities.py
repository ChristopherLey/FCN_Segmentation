import os
import re
from pathlib import Path
from typing import Optional
from typing import Tuple


def convert_dict_2_html(dictionary):
    html = "<table width=100%>\n"
    for key, value in dictionary.items():
        if isinstance(value, dict):
            html += f"<tr>\n<td>{key}</td>\n<td> </td>\n</tr>\n"
            for key2, value2 in value.items():
                html += f"<tr>\n<td> </td>\n<td>{key2}</td>\n<td>{value2}</td>\n</tr>\n"
        else:
            html += f"<tr>\n<td>{key}</td>\n<td>{value}</td>\n</tr>\n"
    html += "</table>\n"
    return "".join(html)


def test_dicttable():
    test = {"basic": 10, "deeper": {"another_lvl": 20, "something_else": "string"}}
    table = convert_dict_2_html(test)
    print(table)


def get_best_checkpoint(path: Path) -> Tuple[Optional[str], float]:
    checkpoint = None
    best_score = 0.0
    for root, dirs, files in os.walk(path / "checkpoints"):
        for file in files:
            if checkpoint is None:
                checkpoint = file
                best_score = float(
                    re.findall(r"f1_score=(\d+(?:\.\d*)?|\.\d+)", file)[0]
                )
            else:
                new_score = float(
                    re.findall(r"f1_score=(\d+(?:\.\d*)?|\.\d+)", file)[0]
                )
                if new_score > best_score:
                    checkpoint = file
                    best_score = new_score
    return checkpoint, best_score
