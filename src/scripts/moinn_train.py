import os
import torch
from torch.utils.data.sampler import RandomSampler
from torch.optim import Adam
import schnetpack as spk
from schnetpack.nn.cutoff import CosineCutoff

from moinn.nn.clustering_module import Clustering
from moinn.training.loss import clustering_loss_fn, cut_loss, ortho_loss, entropy_loss
from moinn.nn.model import EndToEndModel
from moinn.training.trainer import Trainer
from moinn.utils.parsing import get_parser
from moinn.training.loss import ClusteringLoss
import logging


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


########################################################################################################################
# setup
########################################################################################################################
# parse arguments
parser = get_parser()
args = parser.parse_args()

# load model or create model directory
if os.path.exists(args.model_dir):
    logging.info("loading existing MoINN model...")
else:
    os.makedirs(args.model_dir)

# set seed (random or manually)
if args.manual_seed is not None:
    torch.manual_seed(args.manual_seed)

# store parsing arguments
argparse_dict = vars(args)
jsonpath = os.path.join(args.model_dir, "args.json")
spk.utils.spk_utils.to_json(jsonpath, argparse_dict)


########################################################################################################################
# build models
########################################################################################################################
# load cutoff functions
mincut_cutoff_network = spk.nn.cutoff.get_cutoff_by_string(args.mincut_cutoff_function)
bead_cutoff_network = spk.nn.cutoff.get_cutoff_by_string(args.bead_cutoff_function)

# TODO: cutoff und n_gaussians in arg parser
if args.clustering_mode == "pretrained":
    # load representation model
    schnet_model = torch.load(os.path.join(args.rep_model_dir, "best_model"))
    representation = schnet_model.representation
elif args.clustering_mode == "end_to_end":
    # define representation model
    representation = spk.representation.SchNet(
        n_atom_basis=args.features,
        n_filters=args.features,
        n_interactions=args.interactions,
        cutoff=10.,
        n_gaussians=50,
        cutoff_network=CosineCutoff,
    )
else:
    raise NotImplementedError("clustering mode not implemented, yet. Choose \"pretrained\" or \"end_to_end\"")

# define nn model
clustering_model = Clustering(
    features=args.features,
    max_clusters=args.max_clusters,
    mincut_cutoff_function=mincut_cutoff_network,
    mincut_cutoff_radius=args.mincut_cutoff_radius,
    normalize_mincut_adj=args.normalize_mincut_adj,
    bead_cutoff_function=bead_cutoff_network,
    bead_cutoff_parameters=args.bead_cutoff_parameters,
)
# join models
model = EndToEndModel(representation, clustering_model, args.clustering_mode)

# get dataset
if os.path.basename(args.datapath) == "qm9.db":
    dataset = spk.datasets.QM9(args.datapath)
else:
    dataset = spk.AtomsData(args.datapath)
data_train, data_val, data_test = spk.data.train_test_split(
    dataset, args.split[0], args.split[1], split_file=os.path.join(args.model_dir, "split.npz")
)

# build dataloaders
train_loader = spk.data.AtomsLoader(
    data_train,
    batch_size=args.batch_size,
    sampler=RandomSampler(data_train),
    num_workers=4,
    pin_memory=True,
)
val_loader = spk.data.AtomsLoader(
    data_val, batch_size=args.batch_size, num_workers=2, pin_memory=True
)

# build optimizer
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = Adam(trainable_params, lr=args.lr)

# build loss function
loss_fn = clustering_loss_fn(args)

# set device
device = torch.device("cuda" if args.cuda else "cpu")

# define logger
metrics = [
    ClusteringLoss(cut_loss, "mincut_penalty"),
    ClusteringLoss(ortho_loss, "ortho_penalty", args=args),
    ClusteringLoss(entropy_loss, "entropy")
]
logger = spk.train.TensorboardHook(
    os.path.join(args.model_dir, "log"),
    metrics,
    every_n_epochs=1,
    log_histogram=True,
)
schedule = spk.train.ReduceLROnPlateauHook(
    optimizer=optimizer,
    patience=args.lr_patience,
    factor=args.lr_decay,
    min_lr=args.lr_min,
    window_length=1,
    stop_after_min=True,
)
hooks = [schedule, logger]

trainer = Trainer(
    args.model_dir,
    model,
    loss_fn,
    optimizer,
    train_loader,
    val_loader,
    hooks=[logger],
)

logging.info("Starting training...")
trainer.train(device, n_epochs=args.n_epochs)

print(
    "max mem allocated (MB)", torch.cuda.max_memory_allocated() / 1e6, "\n",
    "max mem cached (MB)", torch.cuda.max_memory_cached() / 1e6
)