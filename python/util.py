import json


def get_model_config(model):
    with open('resources/model{}.json'.format(model)) as json_data:
        d = json.load(json_data)
    return d


def get_model_params(model):
    d = get_model_config(model)
    model_config = d["Model params"]
    return model_config["n_channels"], model_config["depth"], model_config["z_dim"], \
           model_config["n_hid_first"], model_config["lam"], model_config["L"]


def model_path(model):
    d = get_model_config(model)
    return d["Model_path"]


def model_epoch(model):
    d = get_model_config(model)
    return d["Selected_model_epoch"]


def get_training_params(model):
    d = get_model_config(model)
    training_config = d["Training"]
    return training_config["batch_size"], training_config["num_epochs"], training_config["learning_rate"]


