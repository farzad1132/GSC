import csv
import os
from typing import Dict, Union


class ResultsWriter:
    def __init__(
        self,
        filename: str = "",
        header = None,
        extra_keys = (),
        override_existing: bool = True,
    ):
        if header is None:
            header = {}
        filename = os.path.realpath(filename)
        # Create (if any) missing filename directories
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Append mode when not overridding existing file
        mode = "w" if override_existing else "a"
        # Prevent newline issue on Windows, see GH issue #692
        self.file_handler = open(filename, f"{mode}t", newline="\n")
        self.logger = csv.DictWriter(self.file_handler, fieldnames=("r"))

    def write_row(self, epinfo: Dict[str, Union[float, int]]) -> None:
        """
        Close the file handler

        :param epinfo: the information on episodic return, length, and time
        """
        if self.logger:
            self.logger.writerow(epinfo)
            self.file_handler.flush()

    def close(self) -> None:
        """
        Close the file handler
        """
        self.file_handler.close()