
"""
==============================================================
BloodCellAI DatasetInfo
==============================================================
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetInfo:

    id:str

    name:str

    task:str

    annotation:str

    path:Path

    registry:dict

    image_files:list = field(default_factory=list)

    annotation_files:list = field(default_factory=list)

    image_count:int = 0

    annotation_count:int = 0

    parser:str = ""

    adapter:str = ""

    classes:dict = field(default_factory=dict)

    statistics:dict = field(default_factory=dict)

    warnings:list = field(default_factory=list)

    errors:list = field(default_factory=list)

    status:str="Pending"

