import logging
from dataclasses import dataclass, field
from itertools import chain

from mkt.databases.api_schema import GraphQLClient

logger = logging.getLogger(__name__)


STR_DRUG_MOA_QUERY = """
query MechanismsOfActionSectionQuery($chemblId: String!) {
  drug(chemblId: $chemblId) {
    id
    mechanismsOfAction {
      rows {
        mechanismOfAction
        targetName
        targets {
          id
          approvedSymbol
        }
        references {
          source
          urls
        }
      }
      uniqueActionTypes
      uniqueTargetTypes
    }
    parentMolecule {
      id
      name
    }
    childMolecules {
      id
      name
    }
  }
}
"""


@dataclass
class OpenTargets(GraphQLClient):
    """Open Targets GraphQL API client."""

    url: str = "https://api.platform.opentargets.org/api/v4/graphql"
    """Base URL for the Open Targets GraphQL API."""


@dataclass
class OpenTargetsDrugMoA(OpenTargets):
    """Open Targets Drug MoA API client."""

    chembl_id: str = field(kw_only=True)
    """ChEMBL ID of the drug to query for mechanism of action information."""
    query_string: str = STR_DRUG_MOA_QUERY
    """GraphQL query string to retrieve drug mechanism of action information."""
    variables: dict = field(default_factory=lambda: {"chemblId": None})
    """Variables for the GraphQL query retrieve drug mechanism of action information."""

    def __post_init__(self):
        """Initialize the Open Targets Drug MoA API client."""
        self.variables["chemblId"] = self.chembl_id
        super().__post_init__()

    def get_moa(self) -> dict | None:
        """Get the mechanism of action information for the drug."""
        if self.response["data"]["drug"] is None:
            logger.warning("No drug found for ChEMBL ID: %s", self.chembl_id)
            return None
        else:
            list_rows = self.response["data"]["drug"]["mechanismsOfAction"]["rows"]
            list_moa = [[i["approvedSymbol"] for i in j["targets"]] for j in list_rows]
            set_moa = set(chain.from_iterable(list_moa))
            return set_moa
