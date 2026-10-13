"""
Convert the original flax checkpoint (FundamentalFormPredictor.pkl) to a torch checkpoint (.pt).
Needs jax + flax (only for this conversion); the converted file loads with torch alone.

	python -m points_as_tori.convert_checkpoint \
		ext/points-as-tori/src/pointsastori/models/FundamentalFormPredictor.pkl \
		points_as_tori/models/FundamentalFormPredictor.pt
"""

import argparse

import torch

from .network import convert_flax_checkpoint


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('src', help='flax .pkl checkpoint')
	parser.add_argument('dst', help='output torch .pt checkpoint')
	args = parser.parse_args()

	saved = convert_flax_checkpoint(args.src)
	torch.save(saved, args.dst)
	n_params = sum(v.numel() for v in saved['state_dict'].values())
	hyper = {k: v for k, v in saved.items() if k != 'state_dict'}
	print(f'wrote {args.dst}: {n_params} parameters, {hyper}')


if __name__ == '__main__':
	main()
