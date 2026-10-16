"""
PAT — stage TRAIN (new data): train the network (FundamentalFormPredictor) on training files in the original's format,
e.g. those of 02_make_dataset.py, starting from the bundled checkpoint (fine-tuning) or from scratch.

	INPUT   --data: training .npz files (globs), e.g. examples/out/pat_data/datasets/*_train_mode*_*.npz
	        --init: 'bundled' (points_as_tori/models/FundamentalFormPredictor.pt), 'scratch', or a checkpoint path
	        --val: optional validation .npz (same format), evaluated after every epoch
	OUTPUT  <models-dir>/<name>.pt            the final checkpoint (loadable by PointsAsTori(model_path=...))
	        <models-dir>/FundamentalFormPredictor_<epoch>.pt   per-epoch checkpoints (the original's naming)
	        <models-dir>/<name>_history.png   losses per epoch (needs matplotlib)

	python examples/pat/04_train_new.py --data 'examples/out/pat_data/datasets/*_train_mode*_*.npz' \
		--epochs 4 --device cuda:0
	python examples/pat/04_train_new.py --data 'examples/out/pat_data/datasets/*.npz' --init scratch --lr 1e-3 \
		--warmup 0 --batch 32 --epochs 64 --device cuda:0          # as the original's pretraining phase
	python examples/pat/05_fit_trained.py --model examples/out/models/mine.pt       # use it (stage FIT)

Defaults (fine-tuning): learning rate 1e-6, 100 warmup steps, batch 1024 neighborhoods: the original's phases 2-3
settings. Architecture: the bundled checkpoint's (8 layers, 8 heads, embedding 128, MLP 512, k = 64); --init scratch
uses the same. One epoch visits every shape once; each shape gives ceil(points / batch) optimizer steps. The original's
full schedule on its data: 03_train_original.py.
"""

import argparse
import glob
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

import torch  # noqa: E402

import points_as_tori as pat  # noqa: E402
from points_as_tori.infer import DEFAULT_MODEL_PATH  # noqa: E402
from points_as_tori.training.train import TrainFundamentalFormPredictor  # noqa: E402


def main():
	ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
	ap.add_argument('--data', nargs='+', required=True, help='training .npz files (globs)')
	ap.add_argument('--init', default='bundled', help="'bundled', 'scratch' or a checkpoint path (.pt / .pkl)")
	ap.add_argument('--val', default=None, help='validation .npz (optional)')
	ap.add_argument('--epochs', type=int, default=4)
	ap.add_argument('--lr', type=float, default=1e-6)
	ap.add_argument('--warmup', type=float, default=1e2, help='warmup steps of the cosine schedule')
	ap.add_argument('--batch', type=int, default=1024, help='neighborhoods per optimizer step')
	ap.add_argument('--name', default='mine', help='output checkpoint name')
	ap.add_argument('--models-dir', default=os.path.join(REPO, 'examples', 'out', 'models'))
	ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
	ap.add_argument('--original-padding', action='store_true', help="the original's NaN gradients of padded shapes")
	a = ap.parse_args()

	files = sorted({f for pattern in a.data for f in glob.glob(pattern)})
	if not files:
		raise SystemExit(f'no training file matches {a.data}')
	init = {'bundled': DEFAULT_MODEL_PATH, 'scratch': None}.get(a.init, a.init)
	pat.set_device(a.device, torch.float32)  # training runs in float32 (the network's dtype)
	print(f'training on {len(files)} files, init {init or "random"}, {a.epochs} epochs, lr {a.lr}, device {a.device}')
	TrainFundamentalFormPredictor(
		files,
		num_epochs=a.epochs,
		learning_rate=a.lr,
		batch_size=a.batch,
		warmup_steps=a.warmup,
		n_epochs_to_save=1,
		n_shapes_to_save=100,
		load_model_path=init,
		final_save_filepath=a.name,
		models_dir=a.models_dir,
		val_data_filename=a.val,
		plot_filepath=os.path.join(a.models_dir, f'{a.name}_history.png'),
		fix_padding=not a.original_padding,
	)
	print(f'checkpoint: {os.path.join(a.models_dir, a.name + ".pt")}')


if __name__ == '__main__':
	main()
