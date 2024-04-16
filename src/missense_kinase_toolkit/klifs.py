# from bravado.requests_client import RequestsClient
from bravado.client import SwaggerClient


class KLIFS():
    def __init__(self):
        self.url = "https://dev.klifs.net/swagger_v2/swagger.json"
        self._klifs = self.get_klifs_api()

    def get_klifs_api(self):
        klifs_api = SwaggerClient.from_url(
            self.url,
            config={
            "validate_requests": False,
            "validate_responses": False,
            "validate_swagger_spec": False
            }
        )
        return klifs_api

    def get_url(self):
        return self.url

    def get_klifs(self):
        return self._klifs


class HumanKinaseInfo(KLIFS):
    species: str = "Human"
    def __init__(
        self,
        kinase_name: str,
    ) -> None:
        super().__init__()
        self.kinase_name = kinase_name
        self._kinase_info = self.get_kinase_info()

    def get_kinase_info(
        self
    ) -> dict[str, str | int | None]:
        try:
            kinase_info = (
                self._klifs.Information.get_kinase_ID(
                kinase_name=[self.kinase_name], 
                species=self.species)
            .response()
            .result[0]
            )

            list_key = dir(kinase_info)
            list_val = [getattr(kinase_info, key) for key in list_key]

            dict_kinase_info = dict(zip(list_key, list_val))

        except Exception as e:
            print(e)
            list_key = [
                'family', 
                'full_name', 
                'gene_name', 
                'group', 
                'iuphar', 
                'kinase_ID', 
                'name', 
                'pocket', 
                'species', 
                'subfamily', 
                'uniprot'
                ]
            dict_kinase_info = dict(zip(list_key, [None]*len(list_key)))

        return dict_kinase_info

    def get_kinase_name(self):
        return self.kinase_name

    def get_species(self):
        return self.species


# def load_af2active(url, path_save):
#     import os
#     import wget
#     import tarfile

#     if not os.path.exists(path_save):
#         os.makedirs(path_save)
#     else:
#         if os.path.exists(os.path.join(path_save, "Kincore_AF2_HumanCatalyticKinases")):
#             print("File already exists...")
#             return

#     wget.download(url, path_save)

# def get_tdc_dti(source_name="DAVIS"):
#     from tdc.multi_pred import DTI

#     data = DTI(name = source_name)
#     data_davis = DTI(name = 'DAVIS')
#     data_davis.get_data()
#     data_davis.entity1_idx.unique().tolist()

#     data_kiba = DTI(name = 'KIBA')

#     data_kiba.get_data()

#     print(data.label_distribution())
#     data.print_stats()
#     data.entity2_name
#     len(data.entity1_idx.unique())
#     data.entity2_idx.unique()
#     data.
#     split = data.get_split()

