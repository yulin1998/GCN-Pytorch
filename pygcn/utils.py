# # # # # # # #
#@Author      : YuLin
#@Date        : 2022-05-06 15:04:12
#@LastEditors : YuLin
#@LastEditTime: 2022-05-10 15:47:14
#@Description : GCN代码学习
# # # # # # # #
import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)                                               #提取类别：len(classes)=7
    identity = np.identity(len(classes))                                #单位矩阵：shape:[7, 7]
    classes_dict = {c: identity[i, :] for i, c in enumerate(classes)}   #将类标签和单位矩阵行向量对应起来
    labels_onehot = np.array(list(map(classes_dict.get, labels)),       #返回所有标签对应的行向量：shape:[2708, 7]
                             dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)    #shape:[2708, 1433]
    labels = encode_onehot(idx_features_labels[:, -1])                          #shape:[2708, 7]

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}                                 #将value与index对应起来
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),         #shape:[5429, 2]
                                    dtype=np.int32)
    edges_tmp = edges_unordered.flatten()                                       #shape change:[5429, 2] -> [10858,]
    edges = np.array(list(map(idx_map.get, edges_tmp)),                         #将value-value转换成为index-index, shape[5429, 2]
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),  #邻接矩阵（稀疏阵），shape:[2708, 2708]
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    # (1) A = A + A.T * (A.T>A)
    # (2) A = A + A.T * (A.T>A) - A * (A.T>A)
    # 当A为元素只有0，1的矩阵时，（1）==（2）
    # 当A中的元素不止0，1时，（2）是正确的，（1）是错误的
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


if __name__ == '__main__':
    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    print(adj.shape, features.shape, labels.shape)
    print(labels.max().item() + 1)
    pass
