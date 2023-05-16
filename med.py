from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy
from six.moves import range
from sklearn import ensemble, linear_model, preprocessing
from align_and_predict import normalize,cond_ent_for_alignment
from utils import np_ent


def compute_mi_mat(x,y):
    x = normalize(x)
    assert x.ndim == 2
    assert y.ndim == 2
    cost_mat_ = np.empty((x.shape[1],y.shape[1])) # tall thin matrix normally
    for i in range(x.shape[1]):
        factor_x = x[:,i]
        for j in range(y.shape[1]):
            factor_y = y[:,j]
            cost_mat_[i,j] = cond_ent_for_alignment(factor_x,factor_y)

    raw_ents = np.array([np_ent(y[:,j]) for j in range(y.shape[1])])
    return np.transpose(cost_mat_ - raw_ents)

def _compute_med(mus_train, ys_train, mus_test, ys_test, topk):
  """Computes score based on both training and testing codes and factors."""
  scores = {}
  importance_matrix, train_err, test_err = compute_importance_mi(
      mus_train, ys_train, mus_test, ys_test)
  assert importance_matrix.shape[0] == mus_train.shape[0]
  assert importance_matrix.shape[1] == ys_train.shape[0]
  scores["informativeness_train"] = train_err
  scores["informativeness_test"] = test_err
  scores["disentanglement"] = disentanglement(importance_matrix)
  scores["completeness"] = completeness(importance_matrix)

  if topk > 0:
    pick_index = pick_by_dis_per_factor(importance_matrix, topk)
    importance_matrix = importance_matrix[pick_index, :]
    scores[f"top{topk}_disentanglement"] = disentanglement(importance_matrix)
    scores[f"top{topk}_completeness"] = completeness(importance_matrix)

  return scores


def compute_importance_mi(x_train, y_train, x_test, y_test):
  """Compute importance matrix based on Mutual Information and informativeness based on Logistic."""
  num_factors = y_train.shape[0]
  num_codes = x_train.shape[0]
  # Caculate importance by MI like MIG.
  #discretized_mus = utils.make_discretizer(x_train)
  #m = utils.discrete_mutual_info(discretized_mus, y_train)
  m = compute_mi_mat(x_train.T,y_train.T)
  m = m.T
  # m's shape is num_codes x num_factors
  # Norm by factor sum.
  importance_matrix = np.divide(m, m.sum(axis=0))

  train_loss = []
  test_loss = []
  for i in range(num_factors):
    print(i)
    model = linear_model.LogisticRegression(solver='saga',n_jobs=-1,max_iter=10,tol=1e-2)
    # Some case fails to converge, add preprocessing to scale data to zero mean
    # and unit std.
    scaler = preprocessing.StandardScaler().fit(x_train.T)
    x_train_scale = scaler.transform(x_train.T)
    x_test_scale = scaler.transform(x_test.T)
    model.fit(x_train_scale, y_train[i, :])
    #NOTE: Copy and paste from disentangle_lib. It's acctually train acc here
    train_loss.append(np.mean(model.predict(x_train_scale) == y_train[i, :]))
    test_loss.append(np.mean(model.predict(x_test_scale) == y_test[i, :]))
  return importance_matrix, np.mean(train_loss), np.mean(test_loss)

def pick_by_dis_per_factor(importance_matrix, k):
    """ Selection process of Top-k MED. For each factor, selects the most
    k disentangled dimensions. """
    latent_num, factor_num= importance_matrix.shape
    dis_per_code = disentanglement_per_code(importance_matrix)
    sort_index = np.argsort(-1 * dis_per_code)
    factor_per_code = np.argmax(importance_matrix, axis=1)
    factor_dim = [[] for _ in range(factor_num)]
    is_full = [False for _ in range(factor_num)]
    for dim in sort_index:
        cur_factor = factor_per_code[dim]
        if len(factor_dim[cur_factor]) < k:
            factor_dim[cur_factor].append(dim)
        else:
            is_full[cur_factor] = True
        if all(is_full) == True:
            break
    select_index = []
    for fac_d in factor_dim:
        select_index.extend(fac_d)
    return list(set(select_index))

def disentanglement_per_code(importance_matrix):
  """Compute disentanglement score of each code."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                  base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
  """Compute the disentanglement score of the representation."""
  per_code = disentanglement_per_code(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

  return np.sum(per_code*code_importance)


def completeness_per_factor(importance_matrix):
  """Compute completeness of each factor."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                  base=importance_matrix.shape[0])


def completeness(importance_matrix):
  """"Compute completeness of the representation."""
  per_factor = completeness_per_factor(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
  return np.sum(per_factor*factor_importance)
