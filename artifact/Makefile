# make numbers   every derived number (LaTeX macros), table and figure-data file, from the CSVs in data/  -> build/
#                (data/*.csv: CPU runs; data/gpu/*.csv: GPU runs; data/legacy/*.csv: earlier CPU runs, superseded by data/*.csv)
# make check     make numbers, then compare build/ with the reference numbers (reference/)
# make regress   compare your reruns (results/*.csv, from hpc/run.sh and hpc/run_gpu.sh) with the reference measurements (data/)
# Version labels: lib-a ... lib-g stand for the library versions the rows were measured with (lib-g: this repository).
PY ?= python3
CPU_COMMITS ?= lib-b,lib-a
GPU_COMMIT ?= lib-g
export GPU_HEAD ?= ours-auto
export GPU_MORDER ?= auto
.PHONY: numbers check regress clean
numbers:
	@rm -rf build && mkdir -p build/data build/figures build/tables
	$(PY) scripts/csv_to_tex.py data/legacy build > build/csv_to_tex.log
	$(PY) scripts/cpu_data.py data build $(CPU_COMMITS) > build/cpu_data.log
	$(PY) scripts/gpu_data.py data/gpu data build "$(GPU_COMMIT)" $(CPU_COMMITS) > build/gpu_data.log
	$(PY) scripts/fig_apsp_combined.py data build/figures "$(GPU_COMMIT)" $(CPU_COMMITS) > build/fig_apsp_combined.log
	$(PY) scripts/table_graphs.py > build/tables/tab-graphs.tex
	@echo "numbers in build/data/{macros,results,gpu_results}.tex; tables in build/tables; figure data in build/figures, build/data"
check: numbers
	$(PY) scripts/compare_macros.py reference/data/gpu_results.tex build/data/gpu_results.tex
	$(PY) scripts/compare_macros.py reference/data/results.tex build/data/results.tex
	$(PY) scripts/compare_macros.py reference/data/macros.tex build/data/macros.tex
regress:
	$(PY) scripts/regress.py
clean:
	rm -rf build
