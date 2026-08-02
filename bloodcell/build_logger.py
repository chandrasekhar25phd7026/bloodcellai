
"""
==============================================================
BloodCellAI Build Logger
==============================================================
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
from datetime import datetime


@dataclass
class BuildRecord:

    dataset: str
    image: str
    status: str
    objects: int
    elapsed: float
    message: str
    timestamp: str


class BuildLogger:

    def __init__(self):

        self.records = []

    def success(self,
                dataset,
                image,
                objects,
                elapsed):

        self.records.append(

            BuildRecord(

                dataset=dataset,

                image=image,

                status="SUCCESS",

                objects=objects,

                elapsed=elapsed,

                message="OK",

                timestamp=str(datetime.now())

            )

        )

    def failure(self,
                dataset,
                image,
                message,
                elapsed=0):

        self.records.append(

            BuildRecord(

                dataset=dataset,

                image=image,

                status="FAILED",

                objects=0,

                elapsed=elapsed,

                message=message,

                timestamp=str(datetime.now())

            )

        )

    def dataframe(self):

        return pd.DataFrame(

            [asdict(r) for r in self.records]

        )

    def export(self, folder):

        folder = Path(folder)

        folder.mkdir(parents=True, exist_ok=True)

        df = self.dataframe()

        df.to_csv(folder / "build_log.csv", index=False)

        df.to_excel(folder / "build_log.xlsx", index=False)

        return df
