import time
import numpy as np
import matplotlib.pyplot as plt

def matrix_multiply(A, B):
    A_shape = A.shape
    B_shape = B.shape
    rows_A = A_shape[0]
    cols_A = A_shape[1]
    rows_B = B_shape[0]
    cols_B = B_shape[1]
    assert cols_A == rows_B
    C = np.zeros((rows_A, cols_B))
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(rows_B):
                C[i][j] += A[i][k] * B[k][j]
    return C

def matrix_multiply_blocked(A, B, BLOCK_SIZE):
    M, K = A.shape
    K, N = B.shape
    C = np.zeros((M, N), dtype=np.float32)
    for m in range(0, M, BLOCK_SIZE):
        for n in range(0, N, BLOCK_SIZE):
            acc = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float32)
            for k in range(0, K, BLOCK_SIZE):
                a = A[m: m + BLOCK_SIZE, k: k + BLOCK_SIZE]
                b = B[k: k + BLOCK_SIZE, n: n + BLOCK_SIZE]
                acc += np.dot(a, b)
            C[m: m + BLOCK_SIZE, n: n + BLOCK_SIZE] = acc
    return C

def benchmark_matrix_multiplication(sizes, BLOCK_SIZE):
    times_naive = []
    times_blocked = []

    for size in sizes:
        a = np.random.randn(size, size)
        b = np.random.randn(size, size)

        # Naive matrix multiplication
        t1 = time.time()
        c2 = matrix_multiply(a, b)
        t2 = time.time()
        times_naive.append(t2 - t1)

        # Blocked matrix multiplication
        t1 = time.time()
        c3 = matrix_multiply_blocked(a, b, BLOCK_SIZE)
        t2 = time.time()
        times_blocked.append(t2 - t1)

        print('size:{}'.format(size))
        print(np.mean(np.abs(c2.flatten()-c3.flatten())))

    return times_naive, times_blocked

if __name__ == '__main__':
    sizes = [32,64,128, 256,512]  # Different matrix sizes to benchmark
    BLOCK_SIZE = 32

    times_naive, times_blocked = benchmark_matrix_multiplication(sizes, BLOCK_SIZE)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_naive, label='Naive (matrix_multiply)', marker='o')
    plt.plot(sizes, times_blocked, label='Blocked (matrix_multiply_blocked)', marker='o')

    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title('Matrix Multiplication Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()