Steps to reproduce:

obtain the datasets, dsprites.npz,   3dshapes.h5, mpi3d.npz, and place them in a datasets/ directory.
Preprocess 3dshapes and mpi3d with 'python preprocess_3dshapes.py' and 'python preprocess_mpi3d.py'
Train and test the VAE models with 

python main.py --vae {vae-name} --dataset {dataset-name} --zs_combo_vals {a,b}

where a and b are, respectively, the value of the size and shape that are going to be excluded, in the compositional generalization task.
The results will save to the experiments/ directory. For the setting where the testset is selected randomly, use a=b=-1.
