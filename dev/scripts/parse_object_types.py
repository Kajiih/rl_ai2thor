"""Parse the data about object types from the AI2THOR documentation."""

import json
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

url = "https://ai2thor.allenai.org/ithor/documentation/objects/object-types/"
object_types_data_path = Path("src/rl_ai2thor/data/object_types_data.json")

with urllib.request.urlopen(url) as response:
    data = response.read()

soup = BeautifulSoup(data.decode("utf-8"), "html.parser")

data_dict = {}
table_body = soup.find("tbody")
items = table_body.find_all("tr")[1:]


def strip_and_uncapitalize(s: str) -> str:
    """Strip and lower the first character of a string."""
    s = s.strip()
    return s[0].lower() + s[1:] if s else s


for item in items:
    tds = item.find_all("td")
    data_dict[tds[0].text.replace("*", "")] = {
        "scenes": tds[1].text,
        "actionable_properties": [strip_and_uncapitalize(prop_str) for prop_str in tds[2].text.split(",")],
        "materials_properties": [strip_and_uncapitalize(prop_str) for prop_str in tds[3].text.split(",")],
        "compatible_receptacles": [prop_str.strip() for prop_str in tds[4].text.split(",")],
        "contextual_interactions": tds[5].text,
    }

with object_types_data_path.open("w") as f:
    json.dump(data_dict, f, indent=4)
    print(f"Object types data written to {object_types_data_path}")
