import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clustering_mode",
                        help="choose if representations are provided or learned",
                        choices=["pretrained", "end_to_end"],
                        required=True)
    parser.add_argument("--datapath", help="path/to/dataset", type=str)
    parser.add_argument("--model_dir", help="directory/of/nn/model", type=str)
    parser.add_argument("--rep_model_dir", help="directory/of/pretrained/MPNN", type=str)
    parser.add_argument(
        "--features", type=int, help="Size of atom-wise representation", default=128
    )
    parser.add_argument(
        "--interactions", type=int, help="Number of interaction blocks", default=6
    )
    parser.add_argument(
        "--cuda", help="Set flag to use GPU(s) for training", action="store_true"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Mini-batch size for nn (default: %(default)s)",
        default=100,
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Initial learning rate (default: %(default)s)",
        default=1e-3,
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        help="Maximum number of training epochs (default: %(default)s)",
        default=1000,
    )
    parser.add_argument(
        "--split_path", help="Path / destination of npz with data splits", default=None
    )
    parser.add_argument(
        "--split",
        help="Split into [train] [validation] and use remaining for testing",
        type=int,
        nargs=2,
        default=[None, None],
    )
    parser.add_argument(
        "--max_clusters",
        type=int,
        default=30,
        help="Max. number of atom-group types (default: %(default)s)",
    )
    parser.add_argument(
        "--mincut_cutoff_function",
        help="Functional form of the cutoff used in mincut loss",
        choices=["hard", "cosine", "mollifier", "cov_bonds"],
        default="cosine",
    )
    parser.add_argument(
        "--mincut_cutoff_radius",
        type=float,
        nargs='*',
        default=[2.0],
        help="cutoff radius determining minimal adjacency distance (ignored when cov_bonds is used, default: %(default)s)",
    )
    parser.add_argument(
        "--normalize_mincut_adj",
        help="mincut adj matrix is normalized",
        action="store_true"
    )
    parser.add_argument(
        "--bead_cutoff_function",
        help="Functional form of the cutoff used to roughly define beads",
        choices=["cosine", "switch"],
        default="cosine",
    )
    parser.add_argument(
        "--bead_cutoff_parameters",
        type=float,
        nargs="*",
        default=[4.0],
        help="for cosine co-function, this represents a cutoff radius determining maximal adjacency distance\n"
             "for swicth co-function, two parameters must be passed (cut-on, cut-off)\n"
             "(used for bead assignments, default: %(default)s)",
    )
    parser.add_argument(
        "--clustering_tradeoff",
        help="tradeoff-factors for nn loss for entropy loss: [ortho], [entropy]",
        type=float,
        nargs=2,
        default=[1.0, 0.06],
    )
    parser.add_argument(
        "--tradeoff_warmup_epochs",
        help="number of epochs for tradeoff-factors warmup: [cut], [entropy]",
        type=float,
        nargs=2,
        default=[100, 130],
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        help="Epochs without improvement before reducing the learning rate "
        "(default: %(default)s)",
        default=25,
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        help="Learning rate decay (default: %(default)s)",
        default=0.8,
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        help="Minimal learning rate (default: %(default)s)",
        default=1e-5,
    )
    parser.add_argument(
        "--manual_seed",
        type=int,
        help="Specify this to reproduce experiments",
        default=None,
    )
    return parser
