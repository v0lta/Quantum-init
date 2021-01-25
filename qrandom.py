# written by moritz (wolter@cs.uni-bonn.de) at 20/01/2021

import numpy as np
from qiskit import(
    QuantumCircuit,
    execute,
    Aer)
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy


def get_backend(nqubits=5):
    """ Returns a quantum computation backend.

    Args:
        nqubits (int, optional): The number of desired qbits.
                                 Defaults to 5.

    Returns:
        backend (qiskit.providers.ibmq.IBMQBackend): The device to run on.
    """    
    provider = IBMQ.load_account()
    # provider = IBMQ.get_provider(hub='ibm-q')
    # backend = least_busy(
    #    provider.backends(
    #        filters=lambda x: x.configuration().n_qubits >= nqubits
    #        and not x.configuration().simulator
    #        and x.status().operational is True))
    # backend = provider.backends.ibmq_armonk
    backend = Aer.get_backend('qasm_simulator')
    print("least busy backend: ", backend)
    return backend


def get_qrandom(shots, backend, n_qbits):
    """ Get a list of quantum random bits.

    Args:
        shots (int): The number of required circuit measurements.
        backend (IBMQBackend): The device to run on.
        n_qbits (int, optional): Number of available qbits. Defaults to 5.

    Returns:
        [list]: A list with uniformly distributed bits.
    """   
    # Create a Quantum Circuit
    circuit = QuantumCircuit(n_qbits)
    # Add a H gate on qubit 0
    for q in range(n_qbits):
        circuit.h(q)
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

            print('qrnd_job', jobno, 'shots', run_shots, 'missing',
                  shots_missing)

            job = execute(circuit, backend, shots=run_shots,
                          memory=True)
            result = job.result()
            partial_mem = result.get_memory(circuit)
            mem.extend(partial_mem)
            total_shots += run_shots
    return mem


def get_array(shape, n_qbits, backend=Aer.get_backend('qasm_simulator')):
    """Generates an array populated with uniformly distributed
       numbers in [0, 1].

    Args:
        shape (tuple): The desired shape of the array.
        n_qbits: The number of qbits per machine. Defaults to 5.
        backend: The Qiskit backend used the generate the
                 random numbers.
                 Defaults to Aer.get_backend('qasm_simulator').

    Returns:
        [np.array]: An array filled with uniformly distributed numbers.
    """

    int_total = np.prod(shape)
    shot_total = np.ceil((16./n_qbits)*int_total)
    mem = get_qrandom(shots=shot_total, backend=backend,
                      n_qbits=n_qbits)

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

    array = np.array(int_lst[:int_total])
    # move to [0, 1]
    array = array / np.power(2, 16)
    array = np.reshape(array, shape)
    return array


def get_quantum_uniform(shape, a, b,
                        n_qbits=5,
                        backend=Aer.get_backend('qasm_simulator')):
    """ Get a numpy array with quantum uniformly initialized numbers

    Args:
        shape (touple): Desired output array shape
        a (float): The lower bound of the uniform distribution
        b (float): The upper bound of the uniform distribution
        n_qbits (int): The number of qbits to utilize.
        backend (optional): Quiskit backend for number generation.
            Defaults to Aer.get_backend('qasm_simulator').

    Returns:
        uniform (nparray): Array initialized to U(a, b).
    """
    zero_one = get_array(shape, backend=backend, n_qbits=n_qbits)
    spread = np.abs(a) + np.abs(b)
    pos_array = zero_one * spread
    uniform = pos_array + a
    return uniform



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    rnd_array = get_quantum_uniform([128, 128], a=-1., b=1.,
                                    n_qbits=1)
    print(rnd_array.shape)
    print('done')
