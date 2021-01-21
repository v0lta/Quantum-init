# written by moritz (wolter@cs.uni-bonn.de) at 20/01/2021

import numpy as np
from qiskit import(
    QuantumCircuit,
    execute,
    Aer)
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy


def get_backend():
    nqubits = 5
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q')
    backend = least_busy(
        provider.backends(
            filters=lambda x: x.configuration().n_qubits >= nqubits
            and not x.configuration().simulator
            and x.status().operational is True))
    print("least busy backend: ", backend)
    return backend


def get_qrandom(shots, backend):
    # Create a Quantum Circuit
    circuit = QuantumCircuit(5)
    # Add a H gate on qubit 0
    circuit.h(0)
    circuit.h(1)
    circuit.h(2)
    circuit.h(3)
    circuit.h(4)
    # Map the quantum measurement to the classical bits
    # circuit.measure([0,1], [0,1])
    circuit.measure_all()

    max_shots = backend.configuration().max_shots
    if shots < max_shots:
        # Execute the circuit on the qasm simulator
        print('qrnd_job', ' ', 'shots', shots, 'missing', 0)
        job = execute(circuit, backend, shots=shots,
                      memory=True)
        # Grab results from the job
        result = job.result()
        # Return Memory
        mem = result.get_memory(circuit)
    else:
        mem = []
        # split shots over multiple runs.
        runs = int(np.ceil(shots/max_shots))
        total_shots = 0
        for jobno in range(runs):
            shots_missing = int(shots - total_shots)
            if shots_missing > max_shots:
                run_shots = max_shots
            else:
                run_shots = shots_missing

            print('qrnd_job', jobno, 'shots', run_shots, 'missing', shots_missing)

            job = execute(circuit, backend, shots=run_shots,
                          memory=True)
            result = job.result()
            partial_mem = result.get_memory(circuit)
            mem.extend(partial_mem)
            total_shots += run_shots
    return mem


def get_array(shape, backend=Aer.get_backend('qasm_simulator')):
    # todo, bigger ints.
    int_total = np.prod(shape)
    shot_total = np.ceil((16./5.)*int_total)
    mem = get_qrandom(shots=shot_total, backend=backend)

    bits_total = ''
    for m in mem:
        bits_total += m

    int_lst = []
    cint = ''
    for b in bits_total:
        cint += b
        if len(cint) == 16:
            int_lst.append(int(cint, 2))
            cint = ''

    array = np.array(int_lst)
    # move to [0, 1]
    array = array / np.power(2, 16)
    array = np.reshape(array, shape)
    return array


def get_quantum_uniform(shape, a, b,
                        backend=Aer.get_backend('qasm_simulator')):
    """ Get a numpy array with quantum uniformly initialized numbers

    Args:
        shape (touple): Desired output array shape
        a (float): The lower bound of the uniform distribution
        b (float): The upper bound of the uniform distribution
        backend (optional): Quiskit backend for number generation.
            Defaults to Aer.get_backend('qasm_simulator').

    Returns:
        uniform (nparray): Array initialized to U(a, b).
    """
    zero_one = get_array(shape, backend)
    spread = np.abs(a) + np.abs(b)
    pos_array = zero_one * spread
    uniform = pos_array + a
    return uniform



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    rnd_array = get_array([512, 512])
    print(rnd_array.shape)
    print('done')
