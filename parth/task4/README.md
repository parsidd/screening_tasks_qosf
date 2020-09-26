This is a README file for Parth's implementation of task4. The codes are in the src folder, and are python scripts, not notebooks. This was done mainly to allow the use of some commandline arguments included in the file, for ease of getting results. The file task4_minimum_eigenvalue.py is the main code and has command line arguments that can be used while calling it:

python task4_minimum_eigenvalue.py [-lin_search] [-tol <tolerance value>] [-N <value for number of points in lin_search] [-ansatz <1 or 2>]

Any incorrect options will result in the above line being displayed.

As an example, to not use the optimiser and just search through the range of 0 to 2pi with 100 points, and use the second ansatz(refer to report for ansatzes):

python task4_minimum_eigenvalue.py -lin_search -N 100 -ansatz 2

The other file, task4_2_vparams is a file that uses 2 variational parameters instead, and is more like a bonus to the main task. This doesn't have any command line arguments though.
