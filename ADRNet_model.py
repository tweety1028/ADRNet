import torch.nn.functional as F
import torch.nn as nn
import scipy.io as sio

from data_loader import *
import model
import utils
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class ADRNet(object):

    def __init__(self, config, logger, running_cnt):

        self.running_cnt = running_cnt

        self.alpha_train = config.alpha_train
        self.beta_train = config.beta_train
        self.alpha_test = config.alpha_test
        self.beta_test = config.beta_test
        self.dataset = config.dataset

        self.config = config
        self.logger = logger

        self.dropout = config.dropout
        self.EPOCHS = config.epochs
        self.WU_EPOCHS = config.warmup_epochs
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.category = config.category
        self.fmri_hidden_dim = config.fmri_hidden_dim
        self.dti_hidden_dim = config.dti_hidden_dim
        self.fusion_dim = config.fusion_dim
        self.ANCHOR = config.ANCHOR
        self.input_dim_F = config.input_dim_F
        self.input_dim_D = config.input_dim_D
        self.output_dim_FD = config.output_dim_FD

        (
            complete_data,
            train_missed_data,
            test_missed_data,
            train_indices,
            test_indices,
        ) = load_data(
            self.dataset,
            self.input_dim_F,
            self.input_dim_D,
            self.output_dim_FD,
            self.alpha_train,
            self.beta_train,
            self.alpha_test,
            self.beta_test,
        )

        ########################################### data ###########################################
        self.train_data = [complete_data["F_tr"], complete_data["D_tr"]]
        self.train_labels = complete_data["L_tr"].numpy()

        self.test_data = [
            complete_data["F_te"],
            complete_data["D_te"],
        ]

        self.train_indices = train_indices
        self.test_indices = test_indices

        # train missed data
        self.train_dual_data = [
            train_missed_data["I_dual_fmri"],
            train_missed_data["I_dual_dti"],
        ]
        self.train_dual_labels = train_missed_data["I_dual_label"]

        self.train_only_fmri = train_missed_data["I_ofmri"]
        self.train_only_fmri_labels = train_missed_data["I_ofmri_label"]

        self.train_only_dti = train_missed_data["I_odti"]
        self.train_only_dti_labels = train_missed_data["I_odti_label"]
        self.d_fmri = self.train_only_fmri.size(1)
        self.d_dti = self.train_only_dti.size(1)

        # test missed data
        self.test_dual_data = [
            test_missed_data["I_dual_fmri"],
            test_missed_data["I_dual_dti"],
        ]
        self.test_only_fmri = test_missed_data["I_ofmri"]
        self.test_only_dti = test_missed_data["I_odti"]

        self.test_labels = torch.cat(
            (
                test_missed_data["I_dual_label"],
                test_missed_data["I_ofmri_label"],
                test_missed_data["I_odti_label"],
            )
        ).numpy()

        self.train_nums = self.train_labels.shape[0]
        self.train_dual_nums = self.train_dual_data[0].size(0)
        self.train_only_fmri_nums = self.train_only_fmri.size(0)
        self.train_only_dti_nums = self.train_only_dti.size(0)
        assert self.train_nums == (
            self.train_dual_nums + self.train_only_fmri_nums + self.train_only_dti_nums
        )

        self.batch_dual_size = math.ceil(self.batch_size * (1 - self.alpha_train))
        self.batch_fmri_size = math.floor(
            self.batch_size * self.alpha_train * self.beta_train
        )
        self.batch_dti_size = (
            self.batch_size - self.batch_dual_size - self.batch_fmri_size
        )
        assert self.batch_dti_size >= 0

        self.fmri_dim = self.train_data[0].size(1)
        self.dti_dim = self.train_data[1].size(1)
        self.num_classes = self.train_labels.shape[1]

        #################################### model define ##########################################
        # specific hash function
        self.fmri_mlp_enc = model.MLP(
            units=[self.fmri_dim, self.fmri_hidden_dim, self.fusion_dim]
        )
        self.dti_mlp_enc = model.MLP(
            units=[self.dti_dim, self.dti_hidden_dim, self.fusion_dim]
        )
        # TEs for obtain neighbour information
        self.fmri_TEs_enc = model.TransformerEncoder(
            Q_dim=self.dti_dim, K_dim=self.dti_dim, V_dim=self.fmri_dim
        )
        self.dti_TEs_enc = model.TransformerEncoder(
            Q_dim=self.fmri_dim, K_dim=self.fmri_dim, V_dim=self.dti_dim
        )
        # missing data generator
        self.fmri_ffn_enc = model.FFNGenerator(
            input_dim=self.dti_dim, output_dim=self.fmri_dim
        )
        self.dti_ffn_enc = model.FFNGenerator(
            input_dim=self.fmri_dim, output_dim=self.dti_dim
        )
        # final hash function
        self.fusion_model = model.Fusion(
            fusion_dim=self.fusion_dim, category=self.category
        )
        # final hash function
        # self.fusion_model2 = model.Fusion2(fusion_dim=self.fusion_dim)

        if torch.cuda.is_available():
            self.fmri_mlp_enc.cuda(), self.dti_mlp_enc.cuda()
            self.fmri_TEs_enc.cuda(), self.dti_TEs_enc.cuda()
            self.fmri_ffn_enc.cuda(), self.dti_ffn_enc.cuda()
            self.fusion_model.cuda()  # , self.fusion_model2.cuda()

        ################################# criterion define #########################################
        self.reconstruction_criterion = nn.MSELoss()
        self.label_criterion = nn.CrossEntropyLoss()
        ################################# optimizer define #########################################
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.fmri_mlp_enc.parameters(), "lr": self.lr},
                {"params": self.dti_mlp_enc.parameters(), "lr": self.lr},
                {"params": self.fmri_ffn_enc.parameters(), "lr": self.lr},
                {"params": self.dti_ffn_enc.parameters(), "lr": self.lr},
                {"params": self.fusion_model.parameters(), "lr": self.lr},
                # {"params": self.fusion_model2.parameters(), "lr": self.lr},
            ]
        )

        self.TEs_optimizer = torch.optim.Adam(
            [
                {"params": self.fmri_TEs_enc.parameters(), "lr": self.lr},
                {"params": self.dti_TEs_enc.parameters(), "lr": self.lr},
            ]
        )

        ################################## anchor extract ##########################################
        self.anchor_nums = config.anchor_nums
        if self.anchor_nums > self.train_dual_nums:
            self.logger.critical(
                "The anchor number is large than the number of dual samples."
            )
            self.anchor_nums = self.train_dual_nums

        # anchor must belong to train dual data!
        self.anchor_idx = np.random.permutation(self.train_dual_nums)[
            : self.anchor_nums
        ]

        self.fmri_anchor = self.train_dual_data[0][self.anchor_idx, :].cuda()
        self.dti_anchor = self.train_dual_data[1][self.anchor_idx, :].cuda()
        self.anchor_label = self.train_dual_labels[self.anchor_idx, :].cuda()

        self.test_code = None

        ################################# hyper-parameter define ####################################
        self.param_neighbour = config.param_neighbour
        self.param_sim = config.param_sim
        self.param_label = config.param_label

        self.batch_count = int(math.ceil(self.train_nums / self.batch_size))

        ################################# metric value ####################################
        self.average_map = 0

    def warmup(self):
        self.fmri_TEs_enc.train(), self.dti_TEs_enc.train()

        self.train_loader = data.DataLoader(
            TrainCoupledData(
                self.train_dual_data[0], self.train_dual_data[1], self.train_dual_labels
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        for epoch in range(self.WU_EPOCHS):
            for batch_idx, (fmri_forward, dti_forward, label) in enumerate(
                self.train_loader
            ):
                fmri_forward = fmri_forward.cuda()
                dti_forward = dti_forward.cuda()
                label = label.cuda()
                self.TEs_optimizer.zero_grad()

                graph = utils.GEN_S_GPU(label, self.anchor_label)

                fmri_neighbour = self.fmri_TEs_enc(
                    dti_forward, self.dti_anchor, self.fmri_anchor, graph
                )
                dti_neighbour = self.dti_TEs_enc(
                    fmri_forward, self.fmri_anchor, self.dti_anchor, graph
                )

                LOSS = self.reconstruction_criterion(
                    fmri_neighbour, fmri_forward
                ) + self.reconstruction_criterion(dti_neighbour, dti_forward)

                LOSS.backward(retain_graph=True)
                self.TEs_optimizer.step()

                if batch_idx == 0:
                    self.logger.info(
                        "[%4d/%4d] (Warm-up) Loss: %.4f"
                        % (epoch + 1, self.WU_EPOCHS, LOSS.item())
                    )

    def train(self):
        self.fmri_mlp_enc.train(), self.dti_mlp_enc.train()
        self.fmri_ffn_enc.train(), self.dti_ffn_enc.train()
        self.fmri_TEs_enc.train(), self.dti_TEs_enc.train()
        self.fusion_model.train()  # , self.fusion_model2.train()

        dual_idx = np.arange(self.train_dual_nums)
        ofmri_idx = np.arange(self.train_only_fmri_nums)
        odti_idx = np.arange(self.train_only_dti_nums)

        for epoch in range(self.EPOCHS):

            np.random.shuffle(dual_idx)
            np.random.shuffle(ofmri_idx)
            np.random.shuffle(odti_idx)

            for batch_idx in range(self.batch_count):
                small_dual_idx = dual_idx[
                    batch_idx
                    * self.batch_dual_size : (batch_idx + 1)
                    * self.batch_dual_size
                ]
                small_ofmri_idx = ofmri_idx[
                    batch_idx
                    * self.batch_fmri_size : (batch_idx + 1)
                    * self.batch_fmri_size
                ]
                small_odti_idx = odti_idx[
                    batch_idx
                    * self.batch_dti_size : (batch_idx + 1)
                    * self.batch_dti_size
                ]

                train_dual_fmri = self.train_dual_data[0][small_dual_idx, :].cuda()
                train_dual_dti = self.train_dual_data[1][small_dual_idx, :].cuda()
                train_dual_labels = self.train_dual_labels[small_dual_idx, :].cuda()

                train_only_fmri = self.train_only_fmri[small_ofmri_idx, :].cuda()
                train_only_fmri_labels = self.train_only_fmri_labels[
                    small_ofmri_idx, :
                ].cuda()

                train_only_dti = self.train_only_dti[small_odti_idx, :].cuda()
                train_only_dti_labels = self.train_only_dti_labels[
                    small_odti_idx, :
                ].cuda()

                loss, accuracy = self.trainstep(
                    train_dual_fmri,
                    train_dual_dti,
                    train_dual_labels,
                    train_only_fmri,
                    train_only_fmri_labels,
                    train_only_dti,
                    train_only_dti_labels,
                )

                if (batch_idx + 1) == self.batch_count:
                    self.logger.info(
                        "[%4d/%4d] Loss: %.4f, Acc: %.4f "
                        % (epoch + 1, self.EPOCHS, loss, accuracy)
                    )

    def trainstep(
        self,
        train_dual_fmri,
        train_dual_dti,
        train_dual_labels,
        train_only_fmri,
        train_only_fmri_labels,
        train_only_dti,
        train_only_dti_labels,
    ):

        self.optimizer.zero_grad()

        dual_cnt = train_dual_labels.size(0)

        fmri_forward = torch.cat([train_dual_fmri, train_only_fmri])
        dti_forward = torch.cat([train_dual_dti, train_only_dti])
        fmri_labels = torch.cat([train_dual_labels, train_only_fmri_labels])
        dti_labels = torch.cat([train_dual_labels, train_only_dti_labels])
        labels = torch.cat(
            [train_dual_labels, train_only_fmri_labels, train_only_dti_labels]
        )

        # construct graph
        fmri_graph = utils.GEN_S_GPU(fmri_labels, self.anchor_label)
        dti_graph = utils.GEN_S_GPU(dti_labels, self.anchor_label)
        graph = utils.GEN_S_GPU(labels, labels)

        ##### Forward

        fmri_feat = self.fmri_mlp_enc(fmri_forward)
        dti_feat = self.dti_mlp_enc(dti_forward)

        #
        fmri_recons = self.fmri_ffn_enc(dti_forward)
        dti_recons = self.dti_ffn_enc(fmri_forward)
        fmri_recons_feat = self.fmri_mlp_enc(fmri_recons)
        dti_recons_feat = self.dti_mlp_enc(dti_recons)

        # obtain the neighbour information
        with torch.no_grad():
            fmri_neighbour = self.fmri_TEs_enc(
                dti_forward, self.dti_anchor, self.fmri_anchor, dti_graph
            )
            dti_neighbour = self.dti_TEs_enc(
                fmri_forward, self.fmri_anchor, self.dti_anchor, fmri_graph
            )

        # get final hash code
        dual_repre = self.fusion_model(fmri_feat[:dual_cnt], dti_feat[:dual_cnt])
        ofmri_repre = self.fusion_model(
            fmri_feat[dual_cnt:], dti_recons_feat[dual_cnt:]
        )
        odti_repre = self.fusion_model(fmri_recons_feat[dual_cnt:], dti_feat[dual_cnt:])

        # dual_repre2 = self.fusion_model2(fmri_feat[:dual_cnt], dti_feat[:dual_cnt])
        # ofmri_repre2 = self.fusion_model2(
        #     fmri_feat[dual_cnt:], dti_recons_feat[dual_cnt:]
        # )
        # odti_repre2 = self.fusion_model2(
        #     fmri_recons_feat[dual_cnt:], dti_feat[dual_cnt:]
        # )
        # total_repre2 = torch.cat([dual_repre2, ofmri_repre2, odti_repre2])
        total_repre = torch.cat([dual_repre, ofmri_repre, odti_repre])
        total_label = (
            torch.argmax(total_repre, dim=1).unsqueeze(1).type(torch.FloatTensor).cuda()
        )

        ##### loss function
        LOSS_label = self.label_criterion(
            total_repre, labels.squeeze(dim=1).type(torch.int64)
        )

        LOSS_sim = self.reconstruction_criterion(total_label.mm(total_label.T), graph)

        # Rewrite to avoid NaN
        if fmri_recons.size(0) != 0 and dti_recons.size(0) == 0:
            LOSS_neighbour = self.reconstruction_criterion(fmri_recons, fmri_neighbour)
        elif fmri_recons.size(0) == 0 and dti_recons.size(0) != 0:
            LOSS_neighbour = self.reconstruction_criterion(dti_recons, dti_neighbour)
        elif fmri_recons.size(0) != 0 and dti_recons.size(0) != 0:
            LOSS_neighbour = self.reconstruction_criterion(
                fmri_recons, fmri_neighbour
            ) + self.reconstruction_criterion(dti_recons, dti_neighbour)
        else:
            LOSS_neighbour = None

        LOSS = LOSS_label * self.param_label + LOSS_sim * self.param_sim

        if LOSS_neighbour != None:
            LOSS = LOSS + LOSS_neighbour * self.param_neighbour

        LOSS.backward(retain_graph=True)
        self.optimizer.step()

        # 计算准确率
        accuracy = accuracy_score(labels.cpu().numpy(), total_label.cpu().numpy())
        return LOSS.item(), accuracy

    def test(self):
        self.logger.info("[TEST STAGE]")
        self.fmri_mlp_enc.eval(), self.dti_mlp_enc.eval()
        self.fmri_ffn_enc.eval(), self.dti_ffn_enc.eval()
        self.fusion_model.eval()

        self.logger.info("test Begin.")
        # test set

        testP = []

        with torch.no_grad():
            dual_fmri_feat = self.fmri_mlp_enc(self.test_dual_data[0].cuda())
            dual_dti_feat = self.dti_mlp_enc(self.test_dual_data[1].cuda())
            dualH = self.fusion_model(dual_fmri_feat, dual_dti_feat)
        testP.append(dualH.data.cpu().numpy())

        with torch.no_grad():
            ofmri_feat = self.fmri_mlp_enc(self.test_only_fmri.cuda())
            ofmri_Gdti = self.dti_ffn_enc(self.test_only_fmri.cuda())
            ofmri_Gdti = self.dti_mlp_enc(ofmri_Gdti)
            ofmriH = self.fusion_model(ofmri_feat, ofmri_Gdti)
        testP.append(ofmriH.data.cpu().numpy())

        with torch.no_grad():
            odti_Gfmri = self.fmri_ffn_enc(self.test_only_dti.cuda())
            odti_Gfmri = self.fmri_mlp_enc(odti_Gfmri)
            odti_feat = self.dti_mlp_enc(self.test_only_dti.cuda())
            odtiH = self.fusion_model(odti_Gfmri, odti_feat)
        testP.append(odtiH.data.cpu().numpy())

        testH = np.concatenate(testP)
        # save_path = "testE.mat"  # Specify the file path for saving
        # sio.savemat(save_path, {"testE": testH})

        self.logger.info("Test End.")

        assert testH.shape[0] == self.test_labels.shape[0]

        # 计算准确率
        predicted_labels = np.argmax(testH, axis=1)
        true_labels = self.test_labels.squeeze(1).astype(np.int64)

        accuracy = accuracy_score(true_labels, predicted_labels)
        save_path = "ADNIROC.mat"  # Specify the file path for saving
        sio.savemat(
            save_path,
            {"true_labels": true_labels, "predicted_labels": testH},
        )
        f1 = f1_score(true_labels, predicted_labels, average="macro")

        testH_onehot = np.eye(3)[predicted_labels]
        auc = roc_auc_score(self.test_labels, testH_onehot, multi_class="ovr")

        self.logger.info(f"Accuracy: {accuracy * 100:.2f}%")
        self.logger.info(f"F1 Score: {f1 * 100:.2f}%")
        self.logger.info(f"AUC: {auc * 100:.2f}%")

        # 计算缺失数据生成器生成的数据与原输入的完整数据之间的余弦相似度
        cosine_similarities = []
        with torch.no_grad():
            # 对 test_dual_data 计算余弦相似度
            dual_index = self.test_indices["dual"]
            for idx in dual_index:
                complete_fmri_feat = self.fmri_mlp_enc(
                    self.test_data[0][idx].unsqueeze(0).cuda()
                )
                complete_dti_feat = self.dti_mlp_enc(
                    self.test_data[1][idx].unsqueeze(0).cuda()
                )

                dual_fmri_recons = self.fmri_ffn_enc(
                    self.test_data[1][idx].unsqueeze(0).cuda()
                )
                dual_fmri_feat = self.fmri_mlp_enc(dual_fmri_recons)

                dual_dti_recons = self.dti_ffn_enc(
                    self.test_data[0][idx].unsqueeze(0).cuda()
                )
                dual_dti_feat = self.dti_mlp_enc(dual_dti_recons)

                dual_fmri_cos_sim = F.cosine_similarity(
                    complete_fmri_feat, dual_fmri_feat, dim=1
                )
                dual_dti_cos_sim = F.cosine_similarity(
                    complete_dti_feat, dual_dti_feat, dim=1
                )
                # cosine_similarities.extend(
                #     [dual_fmri_cos_sim.cpu().numpy(), dual_dti_cos_sim.cpu().numpy()]
                # )

            # 对 test_only_fmri 计算余弦相似度
            ofmri_index = self.test_indices["ofmri"]
            for idx in ofmri_index:
                complete_dti_feat = self.dti_mlp_enc(
                    self.test_data[1][idx].unsqueeze(0).cuda()
                )

                ofmri_recons = self.dti_ffn_enc(
                    self.test_data[0][idx].unsqueeze(0).cuda()
                )
                ofmri_feat = self.dti_mlp_enc(ofmri_recons)
                ofmri_cos_sim = F.cosine_similarity(
                    complete_dti_feat, ofmri_feat, dim=1
                )
                # cosine_similarities.append(ofmri_cos_sim.cpu().numpy())

            # 对 test_only_dti 计算余弦相似度
            odti_index = self.test_indices["odti"]
            for idx in odti_index:
                complete_fmri_feat = self.fmri_mlp_enc(
                    self.test_data[0][idx].unsqueeze(0).cuda()
                )
                odti_recons = self.fmri_ffn_enc(
                    self.test_data[1][idx].unsqueeze(0).cuda()
                )
                odti_feat = self.fmri_mlp_enc(odti_recons)
                odti_cos_sim = F.cosine_similarity(complete_fmri_feat, odti_feat, dim=1)
                cosine_similarities.append(odti_cos_sim.cpu().numpy())

        cosine_similarities = np.concatenate(cosine_similarities)
        average_cosine_similarity = np.mean(cosine_similarities)

        self.logger.info(f"Average Cosine Similarity: {average_cosine_similarity:.4f}")
