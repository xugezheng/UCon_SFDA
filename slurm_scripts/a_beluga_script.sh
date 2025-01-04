# office-31
sbatch --array=0-2 o31_BELUGA_UCon_SFDA.sh amazon dslr
sbatch --array=0-2 o31_BELUGA_autoUCon_SFDA.sh amazon dslr

# office-home
sbatch --array=0-2 oh_BELUGA_UCon_SFDA.sh Pr Rw
sbatch --array=0-2 oh_BELUGA_autoUCon_SFDA.sh Pr Rw