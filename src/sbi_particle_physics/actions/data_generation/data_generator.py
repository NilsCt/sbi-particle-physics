import subprocess
from sbi_particle_physics.config import DATA_DIR, DEFAULT_POINTS_PER_SAMPLE, DEFAULT_SAMPLES_PER_FILE, DEFAULT_PRIOR_LOW, DEFAULT_PRIOR_HIGH, DEFAULT_STRIDE, DEFAULT_PRE_N, DEFAULT_PRERUNS

# I should not forget to activate the conda environment before executing this file

directory = DATA_DIR / "data_3"
start_index = 2500
amount_per_worker = 1
amount_of_workers = 1 # NE PAS DÉPASSER 75

n_samples = 50 # per file
n_points = 10000
prior_low = DEFAULT_PRIOR_LOW[0]
prior_high = DEFAULT_PRIOR_HIGH[0]
stride = DEFAULT_STRIDE
pre_N = DEFAULT_PRE_N
preruns = DEFAULT_PRERUNS

#n_samples = 2 # for tests
#n_points = 10
#prior_low = 3
#prior_high = 5
#stride = 2
#pre_N = 2
#preruns = 2

workers = []
for i in range(amount_of_workers):
    start_index_worker = start_index + i*amount_per_worker
    cmd = [
        "python", "-m", "sbi_particle_physics.actions.worker_data_generator", str(directory), str(start_index_worker), str(amount_per_worker), str(n_samples), str(n_points), str(prior_low), str(prior_high), str(stride), str(pre_N), str(preruns)
    ]

    # execute the workers in parallel (so that they can work at the same time)
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # to cut all outputs
    workers.append(p)

# I have to wait that all workers are finished (otherwise they get killed on start)
# and this allows to kill all the workers with ctrl+c in the terminal
for w in workers:
    w.wait()
print("End of generation")