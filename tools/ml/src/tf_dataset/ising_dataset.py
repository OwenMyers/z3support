
class IsingDataset:

    @staticmethod
    def __new__(cls, train=True, train_percent=80):

        if train:
            return x_train, y_train
        return x_test, y_test
