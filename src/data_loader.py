import pandas as pd
import numpy as np

def data_loader():
    folder = "../data/setting_a/"

    train_bad = pd.read_csv(folder + 'training/bad_training_data_exporting', sep=',', header=None)
    train_good = pd.read_csv(folder + 'training/good_training_data_exporting', sep=',', header=None)

    valid_bad = pd.read_csv(folder + 'valid/bad_valid_data_exporting', sep=',', header=None)
    valid_good = pd.read_csv(folder + 'valid/good_valid_data_exporting', sep=',', header=None)

    test_bad = pd.read_csv(folder + 'test/bad_test_data_exporting', sep=',', header=None)
    test_good = pd.read_csv(folder + 'test/good_test_data_exporting', sep=',', header=None)

    return train_bad.values.astype(np.float), \
           train_good.values.astype(np.float), \
           valid_bad.values.astype(np.float), \
           valid_good.values.astype(np.float), \
           test_bad.values.astype(np.float), \
           test_good.values.astype(np.float)

