import argparse

from missense_kinase_toolkit import config, cbioportal

def parsearg_utils():
    parser = argparse.ArgumentParser(
        description="Get mutations from cBioPortal cohort and instance"
    )

    parser.add_argument(
        "--cohort",
        type=str,
        help="Optional: cBioPortal cohort IDs separated by commas (e.g., `msk_impact_2017` for Zehir, 2017 and `mskimpact` for MSKCC clinical sequencing cohort)",
        default="msk_impact_2017",
    )

    parser.add_argument(
        "--outDir",
        type=str,
        help="Required: Output directory path (str)",
    )

    parser.add_argument(
        "--instance",
        type=str,
        help="Optional: cBioPortal instance (e.g., `cbioportal.mskcc.org`). Default: `cbioportal.org` (str)",
        default="cbioportal.org",
    )

    parser.add_argument(
        "--token",
        type=str,
        default="",
        help="Optional: cBioPortal API token (str)",
    )

    parser.add_argument(
        "--requestsCache",
        type=str,
        default="",
        help="Optional: Requests cache (str)",
    )

    # TODO: add logging functionality
    return parser


def main():
    args = parsearg_utils().parse_args()

    str_studies = args.cohort
    list_studies = str_studies.split(",")
    list_studies = [study.strip() for study in list_studies]

    # required arguments
    config.set_output_dir(args.outDir)
    config.set_cbioportal_instance(args.instance)

    # optional arguments
    try:
        if args.token != "":
            config.set_cbioportal_instance(args.token)
    except AttributeError:
        pass

    try:
        if args.requestsCache != "":
            config.set_cbioportal_instance(args.requestsCache)
    except AttributeError:
        pass

    for study in list_studies:
        cbioportal.get_and_save_cbioportal_cohort(study)
