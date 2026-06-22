import logging
from dataclasses import dataclass

import pandas as pd
from mkt.schema.kinase_schema import KinaseInfo
from mkt.schema.utils import rgetattr

logger = logging.getLogger(__name__)


@dataclass
class PropertyTables:
    """Class to hold the property tables."""

    obj_kinase: KinaseInfo
    """KinaseInfo object from which to extract properties."""
    df_kinhub: pd.DataFrame | None = None
    """Dataframe containing the KinHub information."""
    df_klifs: pd.DataFrame | None = None
    """Dataframe containing the KLIFS information."""
    df_kincore: pd.DataFrame | None = None
    """Dataframe containing the KinCore information."""

    def __post_init__(self):
        """Post-initialization method to extract properties."""
        self.assign_properties()

    def convert_property2dataframe(
        self,
        str_attr: str,
        list_drop: list[str] | None = None,
        list_keep: list[str] | None = None,
    ) -> pd.DataFrame:
        """Convert the properties of the KinaseInfo object to a dataframe.

        Parameters
        ----------
        str_attr : str
            The attribute of the KinaseInfo object to convert to dataframe.
        list_drop : list[str], optional
            The list of attributes to drop from the dataframe, by default None.
            If provided, these attributes will be dropped from the dataframe.
        list_keep : list[str], optional
            The list of attributes to keep in the dataframe, by default None.
            If provided, only these attributes will be kept in the dataframe.

        Returns
        -------
        pd.DataFrame
            The dataframe containing the properties of the KinaseInfo object.
        """
        try:
            obj_temp = rgetattr(self.obj_kinase, str_attr)

            dict_temp = obj_temp.__dict__

            # drop or keep specified attributes
            if list_drop is not None:
                for str_drop in list_drop:
                    del dict_temp[str_drop]

            # keep only specified attributes
            if list_keep is not None:
                dict_temp = {k: dict_temp[k] for k in list_keep if k in dict_temp}

            # convert to dataframe
            df_temp = pd.DataFrame.from_dict(dict_temp, orient="index")
            df_temp.index = df_temp.index.map(lambda x: x.replace("_", " ").upper())
            df_temp.columns = ["Property"]

            return df_temp

        except Exception as e:
            logger.error(f"Error converting properties to dataframe: {e}")
            return None

    def assign_properties(self):
        """Extract the properties from the KinaseInfo object.

        Returns
        -------
        None
            The properties are extracted and stored in the class.
        """
        self.df_kinhub = self.convert_property2dataframe("kinhub")

        self.df_klifs = self.convert_property2dataframe(
            "klifs", list_drop=["pocket_seq"]
        )

        self.df_kincore = self.convert_property2dataframe(
            "kincore.fasta",
            list_keep=["group", "hgnc", "swissprot", "uniprot", "source_file"],
        )
