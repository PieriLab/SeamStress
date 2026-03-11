benzene:
	uv run python benchmark_permutations.py \
		-f data/benzene/spawns \
		-c data/benzene/centroids \
		--master data/benzene/benzene_opt.xyz \
		--allow-reflection

ethylene:
	uv run python benchmark_permutations.py \
		-f data/ethylene/spawns \
		-c data/ethylene/mecis \
		--master data/ethylene/ethylene_opt.xyz \
		--allow-reflection
