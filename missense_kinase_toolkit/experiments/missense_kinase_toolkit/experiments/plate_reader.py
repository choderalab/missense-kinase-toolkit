import xml.etree.ElementTree as ET
from datetime import datetime

# i-control manual: https://bif.wisc.edu/wp-content/uploads/sites/389/2017/11/i-control_Manual.pdf


class Experiment:
    """Class to process Tecan i-control .XML output."""

    def __init__(self, filepath: str) -> None:
        """Initialize Experiment Class object.

        Parameters
        ----------

        Attributes
        ----------

        """
        self.filepath = filepath
        self.measurements = []

        # try except parsing an xml file

        if filepath.lower()[-4:] != ".xml":
            raise TypeError("Filepath does not point to .xml file")

        tree = ET.parse(filepath)
        root = tree.getroot()

        # TODO: check if i-control allows duplicate labels

        for section in root.iter("Section"):
            self.measurements.append(Measurement(section))

    def get_measurement_by_label(self, label: str):

        for i in range(len(self.measurements)):
            if self.measurements[i].label == label:
                return self.measurements[i]


class Measurement:
    """Class to store measurement."""

    def __init__(self, section) -> None:
        """Initialize Measurement Class object.

        Parameters
        ----------

        Attributes
        ----------

        """
        self.section = section
        self.label = section.attrib["Name"]
        self.parameters = {}
        self.time_start = datetime.fromisoformat(section[0].text)
        self.time_end = datetime.fromisoformat(section[-1].text)

        # TODO: add units and other info. as applicable

        for parameters in section.iter("Parameters"):
            for parameter in parameters:
                self.parameters[parameter.attrib["Name"]] = parameter.attrib["Value"]

    def get_duration(self):
        duration = self.time_end - self.time_start
        return duration
