import math
import os

import h5py
import numpy as np
import torch
import torch.utils.data as data
from scipy.io import loadmat, savemat
import model

ROOT = "Data/"
if not os.path.exists(ROOT):
    raise Exception("The ROOT path is error.")

paths = {
    "ADNI": ROOT + "ADNI.mat",
    "Epilepsy": ROOT + "Epilepsy.mat",
}


def load_data(
    DATANAME,
    input_dim_F,
    input_dim_D,
    output_dim,
    alpha_train=0.0,
    beta_train=0.5,
    alpha_test=0.0,
    beta_test=0.5,
):
    data = h5py.File(paths[DATANAME], "r")
    F = torch.Tensor(data["F"][:].T)
    D = torch.Tensor(data["D"][:].T)
    L = torch.Tensor(data["L"][:].T)

    Flat = torch.nn.Flatten(start_dim=1, end_dim=2)
    mlp_D = model.MLP(units=[input_dim_D,  output_dim])
    mlp_F = model.MLP(units=[input_dim_F, output_dim])

    feature_D = torch.Tensor(mlp_D(Flat(D)))
    feature_F = torch.Tensor(mlp_F(Flat(F)))

    # Get number of rows
    numRows = F.size(0)

    # Determine training set size (80%)
    numTrainRows = int(round(0.8 * numRows))

    # Determine test set size (20%)
    numTestRows = numRows - numTrainRows

    # Randomly permute row indices
    randIndices = np.random.permutation(numRows)

    # Split indices for training and test sets
    trainIndices = randIndices[:numTrainRows]
    testIndices = randIndices[numTrainRows:]

    # Create training set
    F_tr = feature_F[trainIndices, :]
    D_tr = feature_D[trainIndices, :]
    L_tr = L[trainIndices, :]

    # Create test set
    F_te = feature_F[testIndices, :]
    D_te = feature_D[testIndices, :]
    L_te = L[testIndices, :]

    # Print dataset sizes to verify
    print(f"Training set size: {F_tr.size(0)}")
    print(f"Test set size: {F_te.size(0)}")

    complete_data = {
        "F_tr": F_tr,
        "D_tr": D_tr,
        "L_tr": L_tr,
        "F_te": F_te,
        "D_te": D_te,
        "L_te": L_te,
    }

    # construct missed data
    train_missed_data, train_indices = construct_missed_data(
        F_tr, D_tr, L_tr, alpha=alpha_train, beta=beta_train
    )
    test_missed_data, test_indices = construct_missed_data(
        F_te, D_te, L_te, alpha=alpha_test, beta=beta_test
    )

    return (
        complete_data,
        train_missed_data,
        test_missed_data,
        train_indices,
        test_indices,
    )


def construct_missed_data(F_tr, D_tr, L_tr, alpha=0.0, beta=0.5):
    number = F_tr.size(0)
    dual_size = math.ceil(number * (1 - alpha))
    only_fmri_size = math.floor(number * alpha * beta)
    only_dti_size = number - dual_size - only_fmri_size
    print(
        "Dual size: %d, Ofmri size: %d, Odti size: %d"
        % (dual_size, only_fmri_size, only_dti_size)
    )

    random_idx = np.random.permutation(number)

    dual_index = random_idx[:dual_size]
    only_fmri_index = random_idx[dual_size : dual_size + only_fmri_size]
    only_dti_index = random_idx[
        dual_size + only_fmri_size : dual_size + only_fmri_size + only_dti_size
    ]

    I_dual_fmri = F_tr[dual_index, :]
    I_dual_dti = D_tr[dual_index, :]
    I_dual_label = L_tr[dual_index, :]

    I_ofmri = F_tr[only_fmri_index, :]
    I_ofmri_label = L_tr[only_fmri_index, :]

    I_odti = D_tr[only_dti_index, :]
    I_odti_label = L_tr[only_dti_index, :]

    _dict = {
        "I_dual_fmri": I_dual_fmri,
        "I_dual_dti": I_dual_dti,
        "I_dual_label": I_dual_label,
        "I_ofmri": I_ofmri,
        "I_ofmri_label": I_ofmri_label,
        "I_odti": I_odti,
        "I_odti_label": I_odti_label,
    }

    indices = {"dual": dual_index, "ofmri": only_fmri_index, "odti": only_dti_index}
    return _dict, indices


class CoupledData(data.Dataset):
    def __init__(self, img_feature, txt_feature):
        self.img_feature = img_feature
        self.txt_feature = txt_feature
        self.length = self.img_feature.size(0)

    def __getitem__(self, item):
        return self.img_feature[item, :], self.txt_feature[item, :]

    def __len__(self):
        return self.length


class TrainCoupledData(data.Dataset):
    def __init__(self, img_feature, txt_feature, label):
        self.img_feature = img_feature
        self.txt_feature = txt_feature
        self.label = label
        self.length = self.img_feature.size(0)

    def __getitem__(self, item):
        return self.img_feature[item, :], self.txt_feature[item, :], self.label[item, :]

    def __len__(self):
        return self.length
