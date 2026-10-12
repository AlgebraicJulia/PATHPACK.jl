#!/bin/bash
# Build FastAPSP (github.com/Liu-xiandong/FastAPSP @ eec2af2) + our driver for sm_100 and sm_120.
# Login node, compile only.  Patch = timers + missing returns + 2 bug fixes (see src/fastapsp/fastapsp.patch).
set -e
command -v module >/dev/null 2>&1 && module load cuda/12.9.1 metis/5.1.0 || true   # (or CUDA 12.9, METIS 5.1.0: set METIS_DIR)
METIS_DIR=${METIS_DIR:-$HPC_METIS_DIR}; CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
B=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # the build root: src/, jobs/, ext/ (fetch_ext.sh), envs/, build/
O=$B/build/fastapsp; S=$O/src
rm -rf $S && mkdir -p $S $O/obj
cp -r $B/ext/FastAPSP/fap $S/
cd $S && patch -p1 < $B/src/fastapsp/fastapsp.patch
ARCH="-gencode arch=compute_100,code=sm_100 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120"
for f in batch_sssp floyd minplus; do
  nvcc -O3 -std=c++14 $ARCH -Xcompiler -fPIC -I$S -c $S/fap/kernel/cuda/$f.cu -o $O/obj/$f.o
done
g++ -O3 -std=c++14 -fopenmp -DWITH_CUDA -I$S -I$METIS_DIR/include -I$CUDA_HOME/include -c $S/fap/fap.cpp -o $O/obj/fap.o
g++ -O3 -std=c++17 -fopenmp -I$S -I$CUDA_HOME/include -c $B/src/fastapsp/fap_driver.cpp -o $O/obj/fap_driver.o
g++ -fopenmp $O/obj/fap.o $O/obj/fap_driver.o $O/obj/batch_sssp.o $O/obj/floyd.o $O/obj/minplus.o -L$METIS_DIR/lib -lmetis -L$CUDA_HOME/lib64 -lcudart_static -ldl -lrt -lpthread -o $O/fap_driver
g++ -O3 -std=c++17 $B/src/fastapsp/mtx2fap.cpp -o $O/mtx2fap
ls -la $O/fap_driver $O/mtx2fap
cuobjdump --list-elf $O/fap_driver | head
# --- multi-GPU (MPI) driver: app/MoreTaskMpiExample.cpp equivalent ---
# The CUDA-aware OpenMPI (cuda/12.9.1 openmpi/5.0.7) needs libhcoll.so.1, which the B200 nodes lack;
# use the plain GCC build of OpenMPI 5.0.7 instead (FastAPSP only uses host MPI_Bcast).
CUDA_INC=$CUDA_HOME/include; CUDA_LIB=$CUDA_HOME/lib64
command -v module >/dev/null 2>&1 && { module purge >/dev/null 2>&1; module load gcc/12.2.0 openmpi/5.0.7 metis/5.1.0; } || true   # (or mpicxx of OpenMPI 5.0.7 on PATH)
mpicxx -O3 -std=c++14 -fopenmp -DWITH_CUDA -I$S -I$METIS_DIR/include -I$CUDA_INC -c $S/fap/fap_more_task_mpi.cpp -o $O/obj/fap_more_task_mpi.mpi.o
mpicxx -O3 -std=c++14 -fopenmp -I$S -I$CUDA_INC -c $B/src/fastapsp/fap_mpi_driver.cpp -o $O/obj/fap_mpi_driver.mpi.o
mpicxx -fopenmp -static-libstdc++ -static-libgcc $O/obj/fap_mpi_driver.mpi.o $O/obj/fap_more_task_mpi.mpi.o $O/obj/fap.o $O/obj/batch_sssp.o $O/obj/floyd.o $O/obj/minplus.o \
  -L$METIS_DIR/lib -lmetis -L$CUDA_LIB -lcudart_static -ldl -lrt -lpthread -o $O/fap_mpi_driver
ls -la $O/fap_mpi_driver
