import subprocess

BASE_START_INDEX = 240 #1700
N_RUNS = 70#70

for i in range(N_RUNS):
    start_index = BASE_START_INDEX + i # attention il faut multiplier si je change amount aussi

    cmd = [
        "nice", "-n", "19",
        "nohup",
        "python", "-m", "sbi_particle_physics.actions.data_generation.data_generation",
        "--start-index", str(start_index),
        "--amount", "1",
        "--directory", "data_6",
        "--n-samples", "1",
        "--n-points", "10000",
        # "--use-imperfections",
    ]

    # Lancement en arrière-plan
    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )

    print(f"Launched job with start-index {start_index}")

# nice -n 19 nohup python nicest_data_generation.py &

# training 6 : ideal final
# training 7 : imperfect final