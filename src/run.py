from src import neural_network
from src import data_loader
from keras.callbacks import EarlyStopping
import numpy as np

x_train_bad, x_train_good, x_valid_bad, x_valid_good, x_test_bad, x_test_good = data_loader.data_loader()


dnn = neural_network.create_dnn()
dnn.compile(optimizer="rmsprop", loss="kld")
callbacks =[ EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min', restore_best_weights=True)]
history = dnn.fit(x_train_bad,x_train_good,
        shuffle=True,
        validation_data = (x_valid_bad, x_valid_good),
        epochs=10000,
        batch_size = 50,
        verbose = 1,
        callbacks = callbacks)


prediction = dnn.predict(x_test_bad)
"Fidelity on testing data (unseen data)", np.average(neural_network.compute_experimental_fidelity_on_test_data(prediction, x_test_good))