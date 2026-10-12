# One task given by run_gpu_split.sh (GPU_TASK, GPU_TASK_ID in the job's environment): the split, parallel submission
# of the GPU targets. Resources per job are set by run_gpu_split.sh.
TASKS=(); TASKS[${GPU_TASK_ID:-0}]="$GPU_TASK"
