# domainnet126
sbatch --array=0-2 domainnet126_NARVAL_UCon_SFDA.sh.sh r p
sbatch --array=0-2 domainnet126_NARVAL_autoUCon_SFDA.sh r p

# oh_partial
sbatch --array=0-2 oh_partial_NARVAL_UCon_SFDA.sh Pr Rw
sbatch --array=0-2 oh_partial_NARVAL_autoUCon_SFDA.sh.sh Pr Rw

# VisDA2017
sbatch --array=0-2 visda2017_NARVAL_UCon_SFDA.sh 
sbatch --array=0-2 visda2017_NARVAL_autoUCon_SFDA.sh 


# VIsDA-RUST
sbatch --array=0-2 visdaRust_NARVAL_UCon_SFDA.sh 
sbatch --array=0-2 visdaRust_NARVAL_autoUCon_SFDA.sh 