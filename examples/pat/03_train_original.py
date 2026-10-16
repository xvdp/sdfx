"""
PAT — stage TRAIN (original data): the original's five-phase training schedule (training/train.py train(), ported in
points_as_tori/training/train.py) on the original's pre-generated datasets.

	INPUT   --data-root: the directory holding the original's `datasets/` folder (pre-generated training data, 6.2 GB:
	        the Google Drive link in ext/points-as-tori/README.md, "Training"), with the paths train() reads,
	        relative to it:
	            datasets/pretraining_blob.npz                  pretraining (one blob)
	            datasets/nice/blobs_train_mode0*.npz           phases 0-4
	            datasets/nice/ABC17_train_mode0_*.npz, ABC2_train_mode0_*.npz, ABC*_train_mode0_*.npz   phases 1-4
	            datasets/simulated/blobs_train_mode*.npz, ABC*_train_mode*.npz                        phase 4
	            datasets/ABC17_validation_3.npz                validation (optional)
	OUTPUT  <models-dir>/FundamentalFormPredictor_pretrained.pt, _phase0.pt ... _phase4.pt (each phase starts from the
	        previous one), per-epoch checkpoints, PlotTrainingHistory.png in <data-root>

	python examples/pat/03_train_original.py --data-root /data/pat --check            # which phase inputs exist
	python examples/pat/03_train_original.py --data-root /data/pat --device cuda:1    # the full schedule
	python examples/pat/03_train_original.py --data-root /data/pat --smoke --device cuda:1   # 1 epoch, 1 file

Phases (epochs, learning rate, batch): pretraining 64, 1e-3, 32 from scratch; 0: 24, 1e-5, 1024; 1: 8, 1e-5; 2: 8,
1e-6; 3: 5, 1e-6; 4: 2, 1e-7 (all batch 1024). Memory: a training step runs the network on every neighborhood of a
shape (2048 points): measured 3.1 GB at 256 points and 5.5 GB at 512 on CPU; ~20 GB at 2048 by linear extrapolation
(not measured). The datasets' layout was not inspected here (the download is not reachable from this environment);
--check lists what is found. Regenerating the data instead: python -m points_as_tori.training.mesh_data (needs the
Thingi10K / ABC downloads and data/blob1024.obj; see the original's training/README).
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
from points_as_tori.training.train import TrainFundamentalFormPredictor, train  # noqa: E402

PHASE_INPUTS = {
	'pretraining': ['datasets/pretraining_blob.npz'],
	'phase 0': ['{nice}/blobs_train_mode0*.npz'],
	'phase 1': ['{nice}/blobs_train_mode0*.npz', '{nice}/ABC17_train_mode0_*.npz'],
	'phase 2': ['{nice}/ABC2_train_mode0_*.npz'],
	'phase 3': ['{nice}/ABC*_train_mode0_*.npz'],
	'phase 4': ['datasets/simulated/blobs_train_mode*.npz', 'datasets/simulated/ABC*_train_mode*.npz'],
	'validation': ['datasets/ABC17_validation_3.npz'],
}


def check(nice: str) -> dict:
	found = {}
	for phase, patterns in PHASE_INPUTS.items():
		files = [f for p in patterns for f in sorted(glob.glob(p.format(nice=nice)))]
		found[phase] = files
		state = f'{len(files)} files' if files else 'MISSING: ' + ', '.join(p.format(nice=nice) for p in patterns)
		print(f'  {phase:12s} {state}')
	return found


def main():
	ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
	ap.add_argument('--data-root', required=True, help='directory containing datasets/')
	ap.add_argument('--dataset-dir', default='datasets/nice', help="the phases' 'nice' folder, relative to --data-root")
	ap.add_argument('--models-dir', default=os.path.join(REPO, 'examples', 'out', 'models_original'))
	ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
	ap.add_argument('--check', action='store_true', help='only list the inputs found per phase')
	ap.add_argument('--smoke', action='store_true', help='1 epoch on the first phase-0 file, from the bundled model')
	ap.add_argument('--batch', type=int, default=1024, help='--smoke batch (<= points per shape)')
	ap.add_argument('--original-padding', action='store_true', help="the original's NaN gradients of padded shapes")
	a = ap.parse_args()

	models_dir = os.path.abspath(a.models_dir)
	os.chdir(a.data_root)  # train() reads 'datasets/...' relative to the working directory, as the original
	print(f'inputs under {os.getcwd()}:')
	found = check(a.dataset_dir)
	if a.check:
		return
	pat.set_device(a.device, torch.float32)
	if a.smoke:
		files = found['phase 0'][:1] or found['phase 3'][:1]
		if not files:
			raise SystemExit('no phase-0 / phase-3 file found')
		TrainFundamentalFormPredictor(
			files, num_epochs=1, learning_rate=1e-6, batch_size=a.batch, warmup_steps=1e2, n_epochs_to_save=1,
			n_shapes_to_save=100, load_model_path=DEFAULT_MODEL_PATH, final_save_filepath='smoke',
			models_dir=models_dir, val_data_filename=(found['validation'] or [None])[0],
		)  # fmt: skip
		return
	missing = [p for p, files in found.items() if not files and p != 'validation']
	if missing:
		raise SystemExit(f'missing inputs for {missing}; see --check')
	train(a.dataset_dir, models_dir, fix_padding=not a.original_padding)
	print(f'checkpoints in {models_dir}; use: python examples/pat/05_fit_trained.py --model {models_dir}/'
		  'FundamentalFormPredictor_phase4.pt')  # fmt: skip


if __name__ == '__main__':
	main()
