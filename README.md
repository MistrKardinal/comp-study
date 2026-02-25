# File Description
### SPDC.ipynb
This is the primary notebook in which simulations based on tensor network methods were performed. The time evolution was computed using the evolute10() routine from the MatrixProductFunctions module. The photon number dynamics in the modes were evaluated using the function calculate_photon_number_in_set_of_solutions() from the same module.

### SPDC_in_photon_basis.ipynb
This notebook contains calculations performed in the Fock basis, intended for comparison with the case of a weak pump mode. The system dynamics were obtained using the function compute_dynamics_in_Fock_basis(). The photon dynamics in the modes were computed using the function calculate_photon_dynamics_in_set_of_solutions_in_Fock_b().

### MatrixProductFunctions.py
This module contains the core functions used for tensor network simulations, including routines for time evolution, photon number evaluation, and reduced density matrix calculations.

### Data_for_article.ipynb
This notebook is used to generate all figures presented in the article. In particular, the data for Fig. 2(c,d) and Fig. 3(a,b) are computed here. Reduced density matrices are calculated using the function find_reduced_density_matrix2() from the MatrixProductFunctions module.

