import subprocess

# I should not forget to activate the conda environment before executing this file

directory = "data/main3"
start_index = 0
amount_per_worker = 1
amount_of_workers = 5 # NE PAS DÉPASSER 75

#n_samples = 500 # per file
#n_points = 100
#prior_low = 3
#prior_high = 5
#stride = 100
#pre_N = 1000
#preruns = 10

n_samples = 2 # per file
n_points = 10
prior_low = 3
prior_high = 5
stride = 2
pre_N = 2
preruns = 2

workers = []
for i in range(amount_of_workers):
    start_index_worker = start_index + i*amount_per_worker
    cmd = [
        "python", "worker_data_generator.py", directory, str(start_index_worker), str(amount_per_worker), str(n_samples), str(n_points), str(prior_low), str(prior_high), str(stride), str(pre_N), str(preruns)
    ]

    # execute the workers in parallel (so that they can work at the same time)
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # to cut all outputs
    workers.append(p)

# I have to wait that all workers are finished (otherwise they get killed on start)
# and this allows to kill all the workers with ctrl+c in the terminal
for w in workers:
    w.wait()