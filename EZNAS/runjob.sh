#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 [command] [job-name]"
    exit 1
fi

COMMAND=$1
JOB_NAME=$2

# Create a temporary SLURM script
TMP_SLURM_SCRIPT="tmp_${JOB_NAME}.slurm"

cat > $TMP_SLURM_SCRIPT <<EOL
#!/bin/bash
#SBATCH -J $JOB_NAME
#SBATCH -o /home/ya255/projects/Hardware-Aware-Automated-Machine-Learning/EZNASV/slurm_logs/${JOB_NAME}_%j.out
#SBATCH -e /home/ya255/projects/Hardware-Aware-Automated-Machine-Learning/EZNASV/slurm_logs/${JOB_NAME}_%j.err
#SBATCH -N 1
#SBATCH --mem=40000
#SBATCH -t 48:00:00
#SBATCH --account=abdelfattah
#SBATCH --partition=abdelfattah
#SBATCH --gres=gpu:1
#SBATCH -n 4

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 
conda activate unr
cd /home/ya255/projects/Hardware-Aware-Automated-Machine-Learning/EZNASV
$COMMAND
EOL

# Submit the temporary script
sbatch --requeue $TMP_SLURM_SCRIPT

# Optionally, remove the temporary script after submission
rm $TMP_SLURM_SCRIPT