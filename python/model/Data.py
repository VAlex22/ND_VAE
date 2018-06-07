class TrainData:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.train_size = 0
        self.validation_size = 0

    def next_batch(self):
        raise NotImplementedError

    def validation_data(self):
        # validation data for early stopping of vae training
        # should include normal data only
        raise NotImplementedError

    def validation_samples(self, size=2):
        raise NotImplementedError

    def test_data(self):
        # test data to compute novelty score on
        raise NotImplementedError
