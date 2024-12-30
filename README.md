# QST_with_MLE_Pauli_measurements
Implementation of Maximum-Likelihood Quantum State Tomography with Pauli measurements for L qubits

# Quantum State Tomography (QST) Using Maximum Likelihood Estimation (MLE) and Pauli Measurements

This repository implements Quantum State Tomography (QST) based on Maximum Likelihood Estimation (MLE) using Pauli measurement operators. The code is written in Python, leveraging PyTorch for efficient matrix operations. The methodology focuses on reconstructing a quantum state's density matrix from simulated measurement data, with two parameterization techniques: Hermitian and Cholesky.

## Key Features

1. **Pauli Measurements**:
   - Generation of tensor products of Pauli operators for multi-qubit systems.
   - Validation of operator normalization and informational completeness.
2. **Density Matrix Reconstruction**:
   - Random density matrix generation.
   - Reconstruction using MLE via Hermitian and Cholesky parameterizations.
3. **Measurement Simulation**:
   - Simulating finite-statistics measurements based on Born's rule.
4. **Validation**:
   - Fidelity and trace distance metrics for comparing the original and reconstructed density matrices.
5. **Special Quantum States**:
   - Implementation of the 3-qubit W state.

---

## Methodology

### 1. Pauli Operators
The set of Pauli operators for \( L \)-qubit systems is generated as tensor products of the single-qubit Pauli matrices \( \{I, X, Y, Z\} \). These operators are normalized to ensure proper scaling:
$P_i \to \frac{P_i}{\sqrt{2^L}}$

### 2. Informational Completeness
The code checks if the set of Pauli operators spans the full Hermitian space of dimension $4^L$ by verifying the rank of the operator matrix.

### 3. Measurement Simulation
Given a density matrix $\hat{\rho}$ and Pauli operators $\{\hat{P}_i\}$, measurement probabilities are computed using Born's rule:
$p_i = \text{Tr}(\hat{\rho} \hat{P}_i)$
Finite-statistics measurements simulate realistic outcomes by generating frequencies for outcomes \(+1\) and \(-1\).

### 4. Maximum Likelihood Estimation
MLE is used to reconstruct the density matrix from simulated measurement data. Two parameterizations are implemented:
- **Cholesky Parameterization**:
  The density matrix is expressed as $\hat{\rho} = \hat{A} \hat{A}^\dagger$, ensuring positivity inherently.

The likelihood function is:
$\mathcal{L} = \sum_i \left( f_i^+ \log(p_i^+) + f_i^- \log(p_i^-) \right)$
where $f_i^+$ and $f_i^-$ are the observed frequencies for outcomes $+1$ and $-1$, respectively.

### 5. Validation Metrics
- **Trace Distance**:
$D(\hat{\rho}_1, \hat{\rho}_2) = \frac{1}{2} \text{Tr}|\hat{\rho}_1 - \hat{\rho}_2|$

### 6. Special Quantum States
The code includes the implementation of the 3-qubit W state:
$|W\rangle = \frac{1}{\sqrt{3}} \left( |001\rangle + |010\rangle + |100\rangle \right)$
Its density matrix is constructed and used as a test case for QST.

---

## Code Structure

### Functions

- **Pauli Operators**:
  - `generate_pauli_operators(L)`: Generate tensor products of Pauli operators.
  - `check_informational_completeness(pauli_operators)`: Check the span of Pauli operators for informational completeness.

- **Density Matrix Handling**:
  - `random_density_matrix(L)`: Generate a random density matrix.
  - `w_state_density_matrix()`: Construct the 3-qubit W state density matrix.
  - `density_matrix_from_params(params, L, parametrization_type)`: Reconstruct a density matrix using Hermitian or Cholesky parameterization.

- **Measurement Simulation**:
  - `simulate_measurements(rho, pauli_operators, N_measurements)`: Simulate measurement frequencies.

- **Optimization and Validation**:
  - `log_likelihood(params, frequencies, pauli_operators, L, parametrization_type)`: Define the likelihood function for MLE.
  - `fidelity(rho1, rho2)`: Compute fidelity between two density matrices.
  - `trace_distance(rho1, rho2)`: Compute trace distance between two density matrices.

- **Utilities**:
  - `print_density_matrix(rho)`: Readable prints real and imaginary parts of a density matrix.

---

## Usage

1. **Generate Pauli Operators**:
   ```python
   pauli_operators, pauli_operator_labels = generate_pauli_operators(L=3)
   check_pauli_operator_sum(pauli_operators)
   check_informational_completeness(pauli_operators)
   ```

2. **Generate W State**:
   ```python
   rho_w = w_state_density_matrix()
   print_density_matrix(rho_w)
   ```

3. **Simulate Measurements**:
   ```python
   frequencies = simulate_measurements(rho_w, pauli_operators, N_measurements=10000)
   ```

4. **Perform MLE Reconstruction**:
   ```python
   result_hermitian = minimize(
       log_likelihood,
       initial_params,
       args=(frequencies, pauli_operators, L, 'hermitian'),
       method="BFGS"
   )
   reconstructed_rho_hermitian = density_matrix_from_params(result_hermitian.x, L, 'hermitian')
   ```

5. **Validate Reconstruction**:
   ```python
   fid = fidelity(rho_w, reconstructed_rho_hermitian)
   tr_dist = trace_distance(rho_w, reconstructed_rho_hermitian)
   print(f"Fidelity: {fid}")
   print(f"Trace Distance: {tr_dist}")
   ```

---

## Results

The code reconstructs density matrices with high fidelity and validates the informational completeness of Pauli measurements. It supports Cholesky parameterizations for flexible and accurate MLE.

---

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- SciPy

---

 
