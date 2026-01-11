import argparse
import time
import utils
from ADRNet_model import *


def _main(config, logger, running_cnt):
    model = ADRNet(config=config, logger=logger, running_cnt=running_cnt)

    logger.info(
        "==========================================================================="
    )
    logger.info("Training stage!")
    start_time = time.time() * 1000
    model.warmup()
    model.train()
    train_time = time.time() * 1000 - start_time
    logger.info("Training time: %.6f" % (train_time / 1000))
    logger.info(
        "==========================================================================="
    )

    logger.info(
        "==========================================================================="
    )
    logger.info("Testing stage!")
    start_time = time.time() * 1000
    model.test()
    test_time = time.time() * 1000 - start_time
    logger.info("Testing time: %.6f" % (test_time / 1000))
    logger.info(
        "==========================================================================="
    )


if __name__ == "__main__":

    utils.seed_setting(seed=3407)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="Epilepsy",
        help="Dataset: ADNI,Epilepsy",
    )
    parser.add_argument(
        "--alpha_train", type=float, default=0.5, help="Missing ratio of train set."
    )
    parser.add_argument(
        "--alpha_test", type=float, default=0.5, help="Missing ratio of test set."
    )
    parser.add_argument("--beta_train", type=float, default=0.5)
    parser.add_argument("--beta_test", type=float, default=0.5)

    parser.add_argument("--input_dim_F", type=int, default=90 * 240, help="ADNI:90*197,Epilespy:90*240")
    parser.add_argument("--input_dim_D", type=int, default=90 * 90)
    parser.add_argument("--output_dim_FD", type=int, default=1024)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=30)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--fmri_hidden_dim", type=int, default=1024)
    parser.add_argument("--dti_hidden_dim", type=int, default=1024)
    parser.add_argument("--fusion_dim", type=int, default=512)
    parser.add_argument("--category", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--anchor_nums", type=int, default=150)

    parser.add_argument(
        "--param_neighbour", type=float, default=0.35, help="Neighbour loss."
    )
    parser.add_argument("--param_sim", type=float, default=0.5, help="Similarity loss.")
    parser.add_argument("--param_label", type=float, default=0.35, help="label loss.")

    parser.add_argument(
        "--ANCHOR", type=str, default="balance", help="Anchor choose!(random/balance)"
    )
    parser.add_argument("--run_times", type=int, default=1)

    logger = utils.logger()

    logger.info(
        "==========================================================================="
    )
    logger.info("Current File: {}".format(__file__))
    config = parser.parse_args()
    utils.log_params(logger, vars(config))
    logger.info(
        "==========================================================================="
    )

    for i in range(config.run_times):
        _main(config=config, logger=logger, running_cnt=i + 1)
