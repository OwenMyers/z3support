cd /home/owen/repos/uml_topological_order/convolutional_unsupervised_learning
python3 generate_samples_z2.py --L 4 --n_configs 3
cd /home/owen/repos/z3support/tools/general/
mv ~/repos/uml_topological_order/convolutional_unsupervised_learning/experiments/raw_configurations/spinConfigs_z2gaugeTheory_L4_N3.txt lattices/
python plot_lattices.py
