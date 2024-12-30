\documentclass{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{hyperref}
\usepackage{graphicx}

\title{Quantum State Tomography (QST) with Maximum Likelihood Estimation (MLE)}
\author{}
\date{}

\begin{document}

\maketitle

\section*{Introduction}
This repository provides a Python implementation of Quantum State Tomography (QST) using Maximum Likelihood Estimation (MLE) with Pauli measurement operators. The code focuses on reconstructing a quantum state's density matrix from simulated measurement data. Key functionalities include Pauli operator generation, measurement simulation, density matrix reconstruction, and validation metrics.

\section*{Key Concepts}

\subsection*{1. Informationally Complete (IC) Measurements}
Informationally Complete (IC) measurements are a set of measurements that allow for the unique determination of any quantum state (density matrix). For a system of $L$ qubits, the density matrix resides in a Hermitian space of dimension $4^L$. To be IC, the measurement operators must span this entire Hermitian space.

\subsection*{2. Pauli Operators and Informational Completeness}
The Pauli matrices for a single qubit are given by:
\[
I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
X = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad
Y = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}, \quad
Z = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}.
\]
For $L$ qubits, the set of Pauli operators is constructed as the tensor product of these matrices:
\[
P_i = \frac{1}{\sqrt{2^L}} (I, X, Y, Z)^{\otimes L}.
\]
These operators are normalized to ensure consistency in the Hilbert-Schmidt inner product.

By verifying the rank of a matrix formed from these operators, the code ensures their informational completeness.

\subsection*{3. Quantum State Tomography with MLE}
QST involves reconstructing the quantum state's density matrix, $\rho$, using measured data. In this implementation, MLE optimizes the likelihood function:
\[
\mathcal{L} = \sum_i \left( f_i^+ \log(p_i^+) + f_i^- \log(p_i^-) \right),
\]
where:
\begin{itemize}
    \item $f_i^+$ and $f_i^-$ are the observed frequencies for measurement outcomes $+1$ and $-1$.
    \item $p_i^+$ and $p_i^-$ are the corresponding Born probabilities, computed as:
\end{itemize}
\[
p_i = \text{Tr}(\rho P_i).
\]

Two parameterizations are implemented:
\begin{enumerate}
    \item \textbf{Cholesky Parameterization}: The density matrix is expressed as $\rho = A A^\dagger$, ensuring positivity inherently.
\end{enumerate}

\subsection*{4. Simulated Frequencies}
The code simulates finite-statistics measurements to emulate realistic data. For each measurement operator, frequencies of outcomes $+1$ and $-1$ are computed, mimicking experimental outcomes.

\section*{Workflow}

\begin{enumerate}
    \item \textbf{Generate Pauli Operators}:
    \begin{verbatim}
    pauli_operators, pauli_operator_labels = generate_pauli_operators(L=3)
    check_informational_completeness(pauli_operators)
    \end{verbatim}
    
    \item \textbf{Generate Quantum States}:
    \begin{itemize}
        \item Random State:
        \begin{verbatim}
        rho = random_density_matrix(L=3)
        \end{verbatim}
        \item W State:
        \begin{verbatim}
        rho = w_state_density_matrix()
        \end{verbatim}
    \end{itemize}
    
    \item \textbf{Simulate Measurements}:
    \begin{verbatim}
    frequencies = simulate_measurements(rho, pauli_operators, N_measurements=10000)
    \end{verbatim}
    
    \item \textbf{Perform Maximum Likelihood Estimation}:
    \begin{verbatim}
    result = minimize(
        log_likelihood,
        initial_params,
        args=(frequencies, pauli_operators, L, 'cholesky'),
        method="BFGS"
    )
    reconstructed_rho = density_matrix_from_params(result.x, L, 'cholesky')
    \end{verbatim}
    
    \item \textbf{Validation}:
    \begin{itemize}
        \item Trace Distance:
        \begin{verbatim}
        tr_dist = trace_distance(rho, reconstructed_rho)
        print(f"Trace Distance: {tr_dist:.6f}")
        \end{verbatim}
        \item Fidelity:
        \begin{verbatim}
        fid = fidelity(rho, reconstructed_rho)
        print(f"Fidelity: {fid:.6f}")
        \end{verbatim}
    \end{itemize}
\end{enumerate}

\section*{Results}

\begin{itemize}
    \item The code demonstrates the convergence of simulated frequencies to Born probabilities as the number of measurements increases.
    \item Reconstruction of the density matrix achieves high fidelity for both random and W states.
    \item Informational completeness of Pauli operators is validated, ensuring reliable QST.
\end{itemize}

\section*{Requirements}

\begin{itemize}
    \item Python 3.8+
    \item PyTorch
    \item NumPy
    \item Matplotlib
    \item SciPy
\end{itemize}

\section*{License}
This project is licensed under the MIT License. Use and modification are allowed for academic and research purposes.

\end{document}
