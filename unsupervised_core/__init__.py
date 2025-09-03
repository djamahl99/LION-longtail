from .dbscan import DBSCAN
from .oyster import OYSTER
from .mfcf import MFCF
from .c_proto_refine import C_PROTO
from .alpha_shape import AlphaShapeMFCF

all_init = {
    "DBSCAN": DBSCAN,
    "OYSTER": OYSTER,
    "MFCF": MFCF,
    "AlphaShapeMFCF": AlphaShapeMFCF
}


all_refine = {
    "C_PROTO": C_PROTO,
}


def compute_outline_box(seq_name, root_path, dataset_cfg):
    suc = None
    if "InitLabelGenerator" in dataset_cfg:
        print("run init outliner")
        init_method = dataset_cfg["InitLabelGenerator"]
        outliner = all_init[init_method](seq_name, root_path, dataset_cfg)
        suc = outliner()
    if "LabelRefiner" in dataset_cfg:
        print("run refiner")
        refine_method = dataset_cfg["LabelRefiner"]
        refiner = all_refine[refine_method](seq_name, root_path, dataset_cfg)
        suc = refiner()
    return suc
