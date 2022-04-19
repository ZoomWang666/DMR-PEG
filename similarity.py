import numpy as np

def matrix_normalize(similarity_matrix):
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    if similarity_matrix.shape[0] == similarity_matrix.shape[1]:
        for i in range(similarity_matrix.shape[0]):
            similarity_matrix[i, i] = 0
        for i in range(200):
            D = np.diag(np.array(np.sum(similarity_matrix, axis=1)).flatten()) # 求得每一行的sum，再使其对角化
            D = np.sqrt(D)
            D[np.isnan(D)] = 0
            D = np.linalg.pinv(D)  # 开方，再取伪逆矩阵
            similarity_matrix = D * similarity_matrix * D
    else:
        for i in range(similarity_matrix.shape[0]):
            if np.sum(similarity_matrix[i], axis=1) == 0:
                similarity_matrix[i] = similarity_matrix[i]
            else:
                similarity_matrix[i] = similarity_matrix[i] / np.sum(similarity_matrix[i], axis=1)
    return similarity_matrix

def get_Cosin_Similarity(interaction_matrix): # 不一定需要方阵，对于M*N的矩阵，看求的是何种对象的similarity，看是否转置矩阵
    X = np.mat(interaction_matrix)
    alpha = np.multiply(X,X).sum(axis=1)
    similarity_matrix = X * X.T / (np.sqrt(alpha * alpha.T))  # 矩阵乘   此步骤之后，得到的是方阵
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    similarity_matrix = similarity_matrix - np.diag(np.diag(similarity_matrix))
    return matrix_normalize(similarity_matrix)


def get_Pearson_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)  # 去中心化版的cosin similarity
    X = X - (X.sum(axis=1) / X.shape[1])  # 每一行向量 行内相加，得到的是行均值。X减去均值，则每一个元素减去均值
    similarity_matrix = get_Cosin_Similarity(X)
    similarity_matrix[np.isnan(similarity_matrix)] = 0  # 缺失值 置0
    similarity_matrix = similarity_matrix - np.diag(np.diag(similarity_matrix))  # 连续使用两个diag，可以得到一个对角阵，除对角线以外的元素均为零
    return matrix_normalize(similarity_matrix)