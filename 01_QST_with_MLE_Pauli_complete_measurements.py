import torch as pt
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np

# Define Pauli matrices
I = pt.eye(2, dtype=pt.cdouble)
X = pt.tensor([[0, 1], [1, 0]], dtype=pt.cdouble)
Y = pt.tensor([[0, -1j], [1j, 0]], dtype=pt.cdouble)
Z = pt.tensor([[1, 0], [0, -1]], dtype=pt.cdouble)

pauli_matrices = [I, X, Y, Z]
pauli_labels = ["I", "X", "Y", "Z"]

def check_informational_completeness(pauli_operators):
    """
    Check if the set of Pauli operators spans the Hermitian space.

    - The space of Hermitian operators for a system of L qubits has dimension 4^L.
    - Flatten each operator into a vector and form a matrix where each row represents a flattened operator.
    - Combine real and imaginary parts of the operators to ensure all degrees of freedom are captured.
    - Compute the rank of this matrix:
        - If the rank equals 4^L, the set of operators spans the Hermitian space.
        - Otherwise, the set is not informationally complete.
    """
    dim = pauli_operators[0].shape[0]
    # Flatten each operator and include real and imaginary parts
    flattened_operators = [pt.cat([P.real.view(-1), P.imag.view(-1)]) for P in pauli_operators]
    operator_matrix = pt.stack(flattened_operators, dim=0)
    
    # Compute the rank of the matrix
    rank = pt.linalg.matrix_rank(operator_matrix).item()
    
    print(f"Number of Hermitian operators (dimension): {dim**2}")
    print(f"Rank of the operator matrix: {rank}")
    
    if rank == dim**2:
        print("The set of Pauli operators is informationally complete.")
    else:
        print("The set of Pauli operators is NOT informationally complete.")



def tensor_product(matrices):
    """Compute the tensor product of a list of matrices."""
    result = matrices[0]
    for matrix in matrices[1:]:
        result = pt.kron(result, matrix)
    return result

def generate_pauli_operators(L):
    """Generate tensor products of Pauli operators for L qubits."""
    pauli_strings = list(itertools.product(pauli_matrices, repeat=L))
    pauli_operators = [tensor_product(p) for p in pauli_strings]
    pauli_operators = [P / pt.sqrt(pt.tensor(2**L, dtype=pt.cdouble)) for P in pauli_operators]
    
    # Generate labels for the Pauli operators
    pauli_operator_labels = []
    for ps in itertools.product(pauli_matrices, repeat=L):
        label = []
        for p in ps:
            for i, matrix in enumerate(pauli_matrices):
                if pt.equal(p, matrix):
                    label.append(pauli_labels[i])
                    break
        pauli_operator_labels.append(" ".join(label))

    return pauli_operators, pauli_operator_labels

def random_density_matrix(L):
    """Generate a random density matrix for L qubits."""
    dim = 2**L
    A = pt.randn((dim, dim), dtype=pt.cdouble)
    rho = A @ A.conj().T
    return rho / pt.trace(rho).real

def simulate_measurements(rho, pauli_operators, N_measurements):
    """Simulate finite-statistics measurements for a given density matrix."""
    frequencies = []
    for P in pauli_operators:
        # Compute the Born probabilities for +1 and -1 outcomes
        prob_plus = (1 + pt.real(pt.trace(rho @ P)).item()) / 2
        prob_minus = 1 - prob_plus
        
        # Simulate measurement outcomes
        counts_plus = pt.sum(pt.rand(N_measurements) < prob_plus).item()
        counts_minus = N_measurements - counts_plus
        
        freq_plus = counts_plus / N_measurements
        freq_minus = counts_minus / N_measurements
        frequencies.append((freq_plus, freq_minus))
    return frequencies

def density_matrix_from_params(params, L, parametrization_type):
    dim = 2**L
    if  parametrization_type == 'cholesky':
        """Construct a density matrix using the Cholesky decomposition."""
        half = dim**2
        real_part = pt.tensor(params[:half], dtype=pt.double).view(dim, dim)
        imag_part = pt.tensor(params[half:], dtype=pt.double).view(dim, dim)
        A = real_part + 1j * imag_part
        rho = A @ A.conj().T
        return rho / pt.trace(rho).real
    else:
        raise ValueError("Unknown parametrization type. 'cholesky'.")

def log_likelihood(params, frequencies, pauli_operators, L, parametrization_type):
    """Compute the negative log-likelihood for the density matrix."""
    rho = density_matrix_from_params(params, L, parametrization_type)
    likelihood = 0
    for (freq_plus, freq_minus), P in zip(frequencies, pauli_operators):
        prob_plus = (1 + pt.real(pt.trace(rho @ P))) / 2  # Keep as tensor
        prob_minus = 1 - prob_plus
        likelihood += freq_plus * pt.log(prob_plus) + freq_minus * pt.log(prob_minus)
    return -likelihood.item()

def fidelity(rho1, rho2):
    """Compute the fidelity between two density matrices."""
    eigvals, eigvecs = pt.linalg.eigh(rho1)
    sqrt_rho1 = eigvecs @ pt.diag(pt.sqrt(pt.clamp(eigvals, min=0))) @ eigvecs.conj().T
    product = sqrt_rho1 @ rho2 @ sqrt_rho1
    product_eigvals = pt.linalg.eigvalsh(product)
    return pt.sum(pt.sqrt(pt.clamp(product_eigvals, min=0))).real

def trace_distance(rho1, rho2):
    """Compute the trace distance between two density matrices."""
    diff = rho1 - rho2
    eigvals = pt.linalg.eigvalsh(diff)
    return 0.5 * pt.sum(pt.abs(eigvals)).real


def w_state_density_matrix():
    """Generate the density matrix for the 3-qubit W state."""
    dim = 2**3  # 3 qubits
    w_state = pt.zeros(dim, dtype=pt.cdouble)
    w_state[1] = 1 / pt.sqrt(pt.tensor(3, dtype=pt.cdouble))
    w_state[2] = 1 / pt.sqrt(pt.tensor(3, dtype=pt.cdouble))
    w_state[4] = 1 / pt.sqrt(pt.tensor(3, dtype=pt.cdouble))
    rho_w = w_state.view(-1, 1) @ w_state.view(1, -1).conj()
    return rho_w

 
def print_density_matrix(rho):
    """Nicely print the real and imaginary parts of a density matrix with two decimal precision."""
    real_part = pt.real(rho)
    imag_part = pt.imag(rho)
    print("Real Part:")
    print(real_part.numpy().round(3))
    print("Imaginary Part:")
    print(imag_part.numpy().round(3))
    
#%%    

# Parameters
L = 3  # Number of qubits


# Generate Pauli operators and labels
pauli_operators, pauli_operator_labels = generate_pauli_operators(L)

# Generate random density matrix
rho = random_density_matrix(L)

 

# Prepare GHZ state
rho = pt.zeros((2**L, 2**L), dtype = pt.complex128)
rho[0,0] = 1
rho[0,-1] = 1
rho[-1, 0] = 1
rho[-1, -1] = 1
rho = 1/2*rho


# Prepare W - state density matrix : only for L = 3
rho = w_state_density_matrix()


N_measurements_list = [1000, 10000, 100000]  # Number of measurements to test

# Compute Born probabilities
born_probabilities = [pt.real(pt.trace(rho @ P)).item() for P in pauli_operators]

# Simulate finite-statistics behavior for varying N_measurements
frequencies_list = [simulate_measurements(rho, pauli_operators, N) for N in N_measurements_list]

# Plot results
plt.figure(figsize=(12, 6))
for i, N_measurements in enumerate(N_measurements_list):
    freq_plus_list = [freq[0] for freq in frequencies_list[i]]
    plt.plot(
        range(len(born_probabilities)), 
        freq_plus_list, 
        label=f"N = {N_measurements} (freq +1)",
        alpha=0.7
    )
plt.plot(
    range(len(born_probabilities)), 
    [(1 + p) / 2 for p in born_probabilities], 
    'k--', 
    label="Born Probabilities (+1)"
)
plt.xlabel("Measurement Operator Index")
plt.ylabel("Probability / Frequency")
plt.title(f"Convergence of Measurement Frequencies to Born Probabilities (L={L})")
plt.legend()
plt.show()

#%%

N_measurements = 10000
# Simulate finite-statistics measurements
frequencies = simulate_measurements(rho, pauli_operators, N_measurements)


# Initial guess for parameters (randomized)
dim = 2**L
half = dim**2
initial_params = pt.cat([pt.randn(half), pt.randn(half)]).tolist()

 
# Cholesky parametrization
result_cholesky = minimize(
    log_likelihood, 
    initial_params, 
    args=(frequencies, pauli_operators, L, 'cholesky'), 
    method="BFGS"
)
reconstructed_rho_cholesky = density_matrix_from_params(result_cholesky.x, L, 'cholesky')

#%%
# Compare with the original density matrix
print("Original Density Matrix:")
print_density_matrix(rho)

 
print("Reconstructed Density Matrix (Cholesky):")
print_density_matrix(reconstructed_rho_cholesky)
 
tr_dist_cholesky = trace_distance(rho, reconstructed_rho_cholesky)
print(f"Trace Distance (Cholesky): {tr_dist_cholesky:.6f}")
