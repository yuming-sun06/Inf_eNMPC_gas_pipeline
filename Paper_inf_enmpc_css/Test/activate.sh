# Activate the existing MPC dependencies in POSIX sh or bash:
# . /home/yumings/test/Test/activate.sh
. /home/yumings/miniconda3/etc/profile.d/conda.sh
conda activate /home/yumings/miniconda3/envs/idaes_env
export PATH="/home/yumings/.idaes/bin:$PATH"
export LD_LIBRARY_PATH="/home/yumings/.idaes/bin:/home/yumings/miniconda3/envs/idaes_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPLCONFIGDIR="/home/yumings/test/Test/.cache/matplotlib"
