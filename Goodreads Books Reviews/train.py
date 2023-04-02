import argparse
import wandb
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score
from wandb.lightgbm import wandb_callback, log_summary

default_config = {"log_preds":False, "boosting_type":"gbdt", "num_leaves":31, "max_depth":-1, "learning_rate":0.1,
     "n_estimators":10, "min_child_samples":20, "subsample":1.0, "colsample_bytree":1.0, "random_state":42,
     "reg_alpha":0.0, "reg_lambda":0.0}

def parse_args():
    # Overriding default argments
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--log_preds', type=bool, default=default_config["log_preds"], help='log model predictions')
    argparser.add_argument('--boosting_type', type=str, default=default_config["boosting_type"], help='booster')
    argparser.add_argument('--num_leaves', type=int, default=default_config["num_leaves"], help='number of leaves')
    argparser.add_argument('--max_depth', type=int, default=default_config["max_depth"], help='maximum tree depth')
    argparser.add_argument('--learning_rate', type=float, default=default_config["learning_rate"], help='learning rate')
    argparser.add_argument('--n_estimators', type=int, default=default_config["n_estimators"], help='number of boosted trees to fit.')
    argparser.add_argument('--min_child_samples', type=int, default=default_config["min_child_samples"], help='minimum child samples')
    argparser.add_argument('--subsample', type=float, default=default_config["subsample"], help='subsample ratio of the training instance')
    argparser.add_argument('--colsample_bytree', type=float, default=default_config["colsample_bytree"], help='subsample ratio of columns when constructing each tree')
    argparser.add_argument('--random_state', type=int, default=default_config["random_state"], help='random state (seed)')
    argparser.add_argument('--reg_alpha', type=float, default=default_config["reg_alpha"], help='L1 regularization term on weights')
    argparser.add_argument('--reg_lambda', type=float, default=default_config["reg_lambda"], help='L2 regularization term on weights')
    args = argparser.parse_args()
    default_config.update(vars(args))
    return


def parse_data():
    # read and prepare data for training
    train = pd.read_csv("train.csv")
    val = pd.read_csv("val.csv")

    train['user_id'] = train['user_id'].astype("category")
    train['review_id'] = train['review_id'].astype("category")
    train['book_id'] = train['book_id'].astype("category")
    train = train.drop("review_text", axis=1)

    val['user_id'] = val['user_id'].astype("category")
    val['review_id'] = val['review_id'].astype("category")
    val['book_id'] = val['book_id'].astype("category")
    val = val.drop("review_text", axis=1)

    return train, val


def evaluate_macroF1_lgb(truth, predictions):
    # this follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483
    pred_labels = predictions.reshape(len(np.unique(truth)),-1).argmax(axis=0)
    f1 = f1_score(truth, pred_labels, average='macro')
    return ('macroF1', f1, True)


def train(config):
    # perform one training iteration
    train, val = parse_data()

    model = lgb.LGBMClassifier(objective='multiclass', boosting_type=config["boosting_type"], num_leaves=config["num_leaves"],
                               max_depth=config["max_depth"], learning_rate=config["learning_rate"],
                               n_estimators=config["n_estimators"], min_child_samples=config["min_child_samples"],
                               subsample=config["subsample"], colsample_bytree=config["colsample_bytree"],
                               random_state=config["random_state"],
                               reg_alpha=config["reg_alpha"],
                               reg_lambda=config["reg_lambda"])

    train_config = model.get_params()
    print("Train configuration:")
    print(train_config)

    run = wandb.init(project="Goodreads Books Reviews", entity="d-a-pop", job_type="training", config=train_config)

    gbm = model.fit(train.drop("rating", axis=1), train["rating"], callbacks=[wandb_callback()],\
                    categorical_feature=["user_id", "book_id", "review_id"], eval_metric=evaluate_macroF1_lgb,\
                    eval_set=[(train.drop("rating", axis=1), train["rating"]),(val.drop("rating", axis=1), val["rating"])],\
                    eval_names=["training", "validation"])

    log_summary(gbm.booster_)

    if config["log_preds"]:
        ypred = model.predict_proba(val.drop("rating", axis=1))
        predictions = val[["review_id", "rating"]]
        predictions["pred"] = np.argmax(ypred, axis=1)
        table = wandb.Table(dataframe=predictions)
        wandb.log({"pred_table": table})

    run.finish()


if __name__ == '__main__':
    parse_args()
    train(default_config)
