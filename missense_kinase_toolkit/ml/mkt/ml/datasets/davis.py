import logging

logger = logging.getLogger(__name__)


def get_tdc_dti(source_name="DAVIS"):
    from tdc.multi_pred import DTI

    data = DTI(name=source_name)
    data_davis = DTI(name="DAVIS")
    data_davis.get_data()
    data_davis.entity1_idx.unique().tolist()

    data_kiba = DTI(name="KIBA")

    data_kiba.get_data()

    print(data.label_distribution())
    data.print_stats()
    data.entity2_name
    len(data.entity1_idx.unique())
    data.entity2_idx.unique()
    split = data.get_split()

    return split, data, data_davis, data_kiba
