import logging
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

# i-control manual: https://bif.wisc.edu/wp-content/uploads/sites/389/2017/11/i-control_Manual.pdf

logger = logging.getLogger(__name__)


@dataclass
class Measurement(ABC):
    """Abstract class for measurements."""

    filepath: str
    """str: Filepath to i-control output .XML."""
    label: str
    """str: Label parameter in i-control."""

    def __post_init__(self):
        self.parse_xml()

    def parse_xml(self):
        try:
            tree = ET.parse(self.filepath)
            root = tree.getroot()
            # TODO: check that i-control does not allow duplicate labels
            self.section_element = root.find(f".//Section[@Name='{self.label}']")
            if self.section_element is None:
                logger.error("Label not found.")
        except ET.ParseError as e:
            logging.error("Could not parse input file", exc_info=e)

    def get_parameter(self, parameter): ...

    @abstractmethod
    def get_data(self) -> int: ...


@dataclass
class Luminescence_Scan(Measurement):
    """Class to parse Luminescence Scan measurement."""

    def get_parameter(self, parameter):
        # TODO
        pass

    def get_data(self, cycle, temperature, well, wavelength):
        data_element = self.section_element.find(
            f".//Data[@Cycle='{str(cycle)}'][@Temperature='{str(temperature)}']"
        )
        if data_element is None:
            logger.error("Data not found.")
        else:
            well_element = data_element.find(f".//Well[@Pos='{well}']")
            scan_element = well_element.find(f".//Scan[@WL='{str(wavelength)}']")
            return int(scan_element.text)
