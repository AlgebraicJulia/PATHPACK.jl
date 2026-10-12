# ============================================================================
#  The ONE file to edit. Everything else reads these settings (defaults in ${...:-...} can also be set in the
#  environment, or in hpc/config.local.sh, which is read last and is not part of the artifact).
# ============================================================================
ACCOUNT=${SLURM_ACCOUNT:-}            # Slurm account (sbatch --account); empty: not passed
QOS=${SLURM_QOS:-}                    # Slurm QOS (sbatch --qos); empty: not passed
PARTITION=${CPU_PARTITION:-cpu}       # CPU partition. The reference runs: AMD EPYC 7702 nodes, 8 NUMA regions x 16 cores
WORK=${WORK:-$REPO/work}              # big scratch space: graph data (~5 GB), Julia depot, baselines
JULIA=${JULIA:-$(command -v julia || echo julia)}   # Julia 1.12 (the reference runs: 1.12.6 and 1.12.7; install with juliaup)

# ---- GPU experiments (hpc/run_gpu.sh, hpc/setup_gpu.sh; the CPU targets do not read these) ----
GPU_PARTITIONS=${GPU_PARTITIONS:-gpu} # GPU partition(s). The reference runs: nodes of 8 x NVIDIA B200 (Xeon Platinum 8570 host)
GPU_GRES_gpu=${GPU_GRES:-b200}        # --gpus=TYPE:N on each partition: GPU_GRES_<partition, with - as _>
GPU_CPUS=${GPU_CPUS:-16}              # host cores per job (a GPU node is not split into NUMA regions; the job's cores
                                      # are whatever Slurm gives, recorded in each row's placement)
GPU_JULIA_THREADS=8                   # julia -t on those cores (16 threads leave no core for the CUDA driver's threads)
# Baselines built from their authors' sources with our patches and drivers (baselines/README.md, hpc/setup_gpu.sh)
ROME_SRC=${ROME_SRC:-}                # a ROME source tree (src/, include/, CMakeLists.txt) with baselines/rome patches
ROME_SRC_INSTR=${ROME_SRC_INSTR:-}    # the same, with phase timers (for the allocation split)
ROME_MODULES=${ROME_MODULES:-}        # ROME's modules/ (METIS) and the Scotch build to link
ROME_SCOTCH=${ROME_SCOTCH:-}
FASTAPSP_BUILD=${FASTAPSP_BUILD:-$WORK/baselines/gpu/build/fastapsp}   # fap_driver, fap_mpi_driver, mtx2fap
GPU_BASELINES_SRC=${GPU_BASELINES_SRC:-$REPO/baselines/gpu}   # drivers and build scripts of the other GPU baselines
GPU_GATE_MTX=${GPU_GATE_MTX:-}        # gpu_tests: integer-weight graphs (coordinate format, both directions stored) and
GPU_GATE_REF=${GPU_GATE_REF:-}        #   reference row hashes of their closures (<graph>.rowhash) from earlier verified runs

if [ -f "$(dirname "${BASH_SOURCE[0]}")/config.local.sh" ]; then source "$(dirname "${BASH_SOURCE[0]}")/config.local.sh"; fi   # per-machine overrides
