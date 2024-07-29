import numpy as np

def compress_matrix(matrix, M):
    N = matrix.shape[0]
    compressed_matrix = np.zeros((N//M+1, N//M+1))
    
    for i in range(N//M+1):
        for j in range(N//M+1):
            compressed_matrix[i, j] = np.mean(matrix[i*M:min((i+1)*M, N), j*M:min((j+1)*M, N)])
    
    return compressed_matrix

# 예제로 사용할 NxN 행렬 생성
N = 7
matrix = np.random.randint(0, 10, (N, N))

# MxM 행렬로 압축
M = 3
compressed = compress_matrix(matrix, M)

print("원래 행렬:")
print(matrix)
print("\n압축된 행렬:")
print(compressed)
