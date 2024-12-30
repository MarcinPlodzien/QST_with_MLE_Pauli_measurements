
# QST_with_MLE_Pauli_measurements

Implementation of Maximum-Likelihood Quantum State Tomography with Pauli measurements for \( L \) qubits.

## Introduction

This repository provides a Python implementation of Quantum State Tomography (QST) using Maximum Likelihood Estimation (MLE) with Pauli measurement operators. The code focuses on reconstructing a quantum state's density matrix from simulated measurement data. Key functionalities include Pauli operator generation, measurement simulation, density matrix reconstruction, and validation metrics.

---

## Key Concepts

### 1. Informationally Complete (IC) Measurements

Informationally Complete (IC) measurements are a set of measurements that allow for the unique determination of any quantum state (density matrix). For a system of \( L \) qubits, the density matrix resides in a Hermitian space of dimension $4^L$. To be IC, the measurement operators must span this entire Hermitian space.

### 2. Pauli Operators and Informational Completeness

The Pauli matrices for a single qubit are given by:

$$
I = egin{bmatrix} 1 & 0 \ 0 & 1 \end{bmatrix}, \quad
X = egin{bmatrix} 0 & 1 \ 1 & 0 \end{bmatrix}, \quad
Y = egin{bmatrix} 0 & -i \ i & 0 \end{bmatrix}, \quad
Z = egin{bmatrix} 1 & 0 \ 0 & -1 \end{bmatrix}.
$$

For \( L \) qubits, the set of Pauli operators is constructed as the tensor product of these matrices:

$$
P_i = rac{1}{\sqrt{2^L}} (I, X, Y, Z)^{\otimes L}.
$$

These operators are normalized to ensure consistency in the Hilbert-Schmidt inner product.

By verifying the rank of a matrix formed from these operators, the code ensures their informational completeness.

### 3. Quantum State Tomography with MLE

QST involves reconstructing the quantum state's density matrix, \( 
ho \), using measured data. In this implementation, MLE optimizes the likelihood function:

$$
\mathcal{L} = \sum_i \left( f_i^+ \log(p_i^+) + f_i^- \log(p_i^-) 
ight),
$$

where:

- $f_i^+$ and $f_i^-$ are the observed frequencies for measurement outcomes \( +1 \) and \( -1 \).
- $p_i^+$ and $p_i^-$ are the corresponding Born probabilities, computed as:

$$
p_i = 	ext{Tr}(
ho P_i).
$$

### 4. Simulated Frequencies

The code simulates finite-statistics measurements to emulate realistic data. For each measurement operator, frequencies of outcomes \( +1 \) and \( -1 \) are computed, mimicking experimental outcomes.

---

## Code Functionality

### Functions

1. **Pauli Operators**:
   - `generate_pauli_operators(L)`: Generates the tensor product of Pauli operators for \( L \)-qubit systems.
   - `check_informational_completeness(pauli_operators)`: Verifies the rank of the operator matrix to ensure informational completeness.

2. **Density Matrix Handling**:
   - `random_density_matrix(L)`: Creates a random density matrix for \( L \)-qubits.
   - `w_state_density_matrix()`: Constructs the density matrix for the 3-qubit W state.
   - `density_matrix_from_params(params, L, parametrization_type)`: Reconstructs a density matrix using the Cholesky parameterization.

3. **Measurement Simulation**:
   - `simulate_measurements(rho, pauli_operators, N_measurements)`: Simulates measurement frequencies based on Born's rule.

4. **Optimization and Validation**:
   - `log_likelihood(params, frequencies, pauli_operators, L, parametrization_type)`: Defines the likelihood function for MLE.
   - `fidelity(rho1, rho2)`: Computes the fidelity between two density matrices.
   - `trace_distance(rho1, rho2)`: Computes the trace distance between two density matrices.

5. **Utilities**:
   - `print_density_matrix(rho)`: Displays the real and imaginary parts of a density matrix in a readable format.

---

## Workflow

1. **Generate Pauli Operators**:
   ```python
   pauli_operators, pauli_operator_labels = generate_pauli_operators(L=3)
   check_informational_completeness(pauli_operators)
   ```

2. **Generate Quantum States**:
   - **Random State**:
     ```python
     rho = random_density_matrix(L=3)
     ```
   - **W State**:
     ```python
     rho = w_state_density_matrix()
     ```

3. **Simulate Measurements**:
   ```python
   frequencies = simulate_measurements(rho, pauli_operators, N_measurements=10000)
   ```

4. **Perform Maximum Likelihood Estimation**:
   ```python
   result = minimize(
       log_likelihood,
       initial_params,
       args=(frequencies, pauli_operators, L, 'cholesky'),
       method="BFGS"
   )
   reconstructed_rho = density_matrix_from_params(result.x, L, 'cholesky')
   ```

5. **Validation**:
   - **Trace Distance**:
     ```python
     tr_dist = trace_distance(rho, reconstructed_rho)
     print(f"Trace Distance: {tr_dist:.6f}")
     ```
   - **Fidelity**:
     ```python
     fid = fidelity(rho, reconstructed_rho)
     print(f"Fidelity: {fid:.6f}")
     ```

6. **Visualization**:
   Simulated measurement frequencies are plotted against Born probabilities to visualize convergence.

---

## Results

- The code demonstrates the convergence of simulated frequencies to Born probabilities as the number of measurements increases.
- Reconstruction of the density matrix achieves high fidelity for both random and W states.
- Informational completeness of Pauli operators is validated, ensuring reliable QST.

---

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- SciPy

---

## License

This project is licensed under the MIT License. Use and modification are allowed for academic and research purposes.
