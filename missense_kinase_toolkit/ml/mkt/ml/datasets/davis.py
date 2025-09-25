import logging
from os import path

import pandas as pd
from mkt.ml.datasets.finetune import FineTuneDataset
from mkt.ml.utils import get_repo_root

logger = logging.getLogger(__name__)

def get_tdc_dti(source_name="DAVIS"):
    from tdc.multi_pred import DTI

    data = DTI(name = source_name)
    data_davis = DTI(name = 'DAVIS')
    data_davis.get_data()
    data_davis.entity1_idx.unique().tolist()

    data_kiba = DTI(name = 'KIBA')

    data_kiba.get_data()

    print(data.label_distribution())
    data.print_stats()
    data.entity2_name
    len(data.entity1_idx.unique())
    data.entity2_idx.unique()
    data.
    split = data.get_split()
