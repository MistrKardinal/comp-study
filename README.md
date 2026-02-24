# The description of code
SPDC.ipynb - основной файл, где проводились вычисления c помощью тензорных сетей.
Временная эволюция была вычислена с помощью скрипта evolute10() из модуля MatrixProductFunctions.
Динамика фотонов в модах была вычислена с помощью функции calculate_photon_number_in_set_of_solutions() из модуля MatrixProductFunctions.

SPDC_in_photon_basis.ipynb - файл, где проводились вычисления в Фоковском базисе для выполнения сравнения с случае со слабой модой накачки.
Функция, с помощью которой были получены решения - compute_dynamics_in_Fock_basis().
Динамика фотонов в модах была вычислена с помощью функции calculate_photon_dynamics_in_set_of_solutions_in_Fock_b()

MatrixProductFunctions - основной модуль с функциями, используемыми при вычислениях с помощью тензорных сетей.

Data_for_article.ipynb - файл, где строятся все графики статьи. В этом же файле вычисляются данные для графика с Fig.2(c), Fig.3(b)
Матрицы плотности вычисляются с помощью функции find_reduced_density_matrix2() из модуля MatrixProductFunctions.

