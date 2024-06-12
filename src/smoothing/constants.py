MIN_STARTS_THRESHOLD = 500

INITIAL_PARAMS = [1, 10]
BOUNDS = [(0.01, None), (0.01, None)]

N_TRIALS = 1000
N_SAMPLES = 100000

COLS_FOR_SMOOTHING = ['dnf', 'scratched', 'vet_scratched', 'badly_beaten', 'breakdown']
DENOMINATORS = {
    'dnf': 'n_starts',
    'scratched': 'n_entries',
    'vet_scratched': 'n_entries',
    'badly_beaten': 'n_starts',
    'breakdown': 'n_starts'
}

DISPLAY_NAMES = {
    'dnf': 'DNF',
    'scratched': 'Scratched',
    'vet_scratched': 'Vet Scratched',
    'badly_beaten': 'Badly Beaten',
    'breakdown': 'Breakdown'
}