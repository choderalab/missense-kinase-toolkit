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
        self.parse_for_section()
        self.parse_for_time()

    def parse_for_section(self):
        try:
            tree = ET.parse(self.filepath)
            root = tree.getroot()
            # TODO: check that i-control does not allow duplicate labels
            self.section_element = root.find(f".//Section[@Name='{self.label}']")
            if self.section_element is None:
                logger.error("Label not found.")
        except ET.ParseError as e:
            logging.error(f"Could not parse input {self.filepath}", exc_info=e)

    def parse_for_time(self):
        self.time_start = datetime.fromisoformat(
            self.section_element.find(".//Time_Start").text
        )
        self.time_end = datetime.fromisoformat(
            self.section_element.find(".//Time_End").text
        )

    def get_parameter(self, parameter: str) -> str:
        # TODO: Add units handling
        parameter_attribute = self.section_element.find(
            f".//Parameter[@Name='{parameter}']"
        ).attrib["Value"]
        return parameter_attribute

    @abstractmethod
    def get_data(self) -> int: ...

    @abstractmethod
    def plot_data(
        self
    ) -> Tuple[matplotlib.figure.Figure, plt.Axes] | None:
    ...


@dataclass
class Luminescence_Scan(Measurement):
    """Class to parse Luminescence Scan measurement."""

    def get_data(self, cycle: int, temperature: int, well: str, wavelength: int):
        data_element = self.section_element.find(
            f".//Data[@Cycle='{str(cycle)}'][@Temperature='{str(temperature)}']"
        )
        if data_element is None:
            logger.error("Data not found.")
        else:
            well_element = data_element.find(f".//Well[@Pos='{well}']")
            scan_element = well_element.find(f".//Scan[@WL='{str(wavelength)}']")
            return int(scan_element.text)

    def plot_data(
        cycle: int
        temperature: int,
        well: str,
        header: str,
        plot_type: str | None = None,
    ):
    if plot_type is None:
        plot_type = "scatter"

    list_plot_type = ["scatter"]
    if plot_type not in list_plot_type:
        logging.error(f"Plot type {plot_type} not yet supported...")
        return None

    if plot_type == "scatter":
        lmin = int(self.get_parameter("Wavelength Start"))
        lmax = int(self.get_parameter("Wavelength End"))
        lstep = int(self.get_parameter("Wavelength Step Size"))
        ll = np.arange(lmin, lmax+lstep, lstep)

        fig, ax = plt.subplots()

        try:
            counter = 0
            for l in ll:
                ax.plot(l, self.get_data(cycle, temperature, well, l), "b.", alpha=0.5)
                counter += 1
            assert counter == int(self.get_parameter("Scan Number"))
            print(f"All {counter} points accounted for")
        except AssertionError as e:
            logging.error(f"{counter} / {int(self.get_parameter('Scan Number'))} points accounted for...")
            return None

        ax.set_box_aspect(1)
        ax.set_title("Water") # make this a parameter?
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Luminescence (RLU)")
        ax.set_xlim([lmin, lmax])
        ax.set_xticks(np.arange(lmin, lmax+50, 50)) # make this a parameter?

        return (fig, ax)
