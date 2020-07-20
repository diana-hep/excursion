#!/bin/bash
#
#SBATCH --job-name=excursionGPU_MES
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --mem=2GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1

#go to directory
cd /scratch/$USER/excursion/excursion

#set environment
module purge
source /home/$USER/.bashrc

module load anaconda3/4.3.1
#module load cuda/9.0.176
#module load pytorch/python3.6/0.3.0_4 
conda activate excursion

#module load cudnn/9.0v7.0.5

#check envirnmt ok
python -c "from __future__ import print_function; import torch; print(torch.version.cuda,torch.cuda.is_available())"
python -c "from __future__ import print_function; import yaml"

#deactivate
#conda deactivate

#end of script


############################################################################


#!/bin/bash
#
#SBATCH --job-name=excursionGPU_MES
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --mem=2GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1

#go to directory
cd /scratch/$USER/excursion/

#set environment
module purge
module load cuda/10.2.89
module load pytorch/python3.6/0.3.0_4

pip install click pyyaml simplejson json scikit-learn gpytorch
pip install -e .


#check envirnmt ok
python -c "from __future__ import print_function; import torch; print(torch.version.cuda,torch.cuda.is_available())"
python -c "from __future__ import print_function; import yaml"

#run job
echo "start job"
cd ./excursion
python3 commandline.py --nupdates 2 --ninit 10 --algorithm_specs "testcases/algorithms/algorithm_specs.yaml" --cuda True  "2Dtoyanalysis" results/

#deactivate
source deactivate

#end of script
