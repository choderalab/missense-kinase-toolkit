import xml.etree.ElementTree as ET


class Experiment:
    """Class to process Tecan i-control .XML output."""

    def __init__(
        self,
        filepath
    ) -> None:
        """Initialize Experiment Class object.

        Parameters
        ----------
        filepath : str
            Filepath to Tecan i-control .XML output.

        Attributes
        ----------
        filepath : str
            Filepath to Tecan i-control .XML output.

        """
        self.filepath = filepath
        self.labels = []

        if filepath.lower()[-4:] != '.xml':
            raise TypeError('Filepath does not point to .xml file')
        
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        for section in root.iter('Section'):
            self.labels.append(section.attrib['Name'])