import lasagne
import theano
import theano.tensor as T
import time
from data_gen import DataGen
from network import NetworkGen


########################################################################################################################
# setup
########################################################################################################################

data_gen = DataGen()

_xs, _ys = data_gen.get_data()

vocab_map = data_gen.get_vocab_map(_xs)

x_trn, y_trn, x_tst, y_tst = data_gen.train_test_split(_xs, _ys, vocab_map)

x = T.imatrix().astype('int32')

network_gen = NetworkGen(input_var=x, sentence_length=data_gen.sentence_length, vocab_size=len(vocab_map))

_network = network_gen.get_network()


########################################################################################################################
# network
#
# Much of the code below is based on http://lasagne.readthedocs.io/en/latest/user/tutorial.html
########################################################################################################################

target_var = T.ivector('targets')


prediction = lasagne.layers.get_output(_network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

params = lasagne.layers.get_all_params(_network, trainable=True)

updates = lasagne.updates.adadelta(loss, params, learning_rate=.9, rho=.95)

test_prediction = lasagne.layers.get_output(_network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                        target_var)
test_loss = test_loss.mean()

test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                  dtype=theano.config.floatX)

train_fn = theano.function([x, target_var], loss, updates=updates)

val_fn = theano.function([x, target_var], [test_loss, test_acc])


########################################################################################################################
# train
########################################################################################################################
num_epochs = 50

for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in data_gen.batch_gen(x_trn, y_trn, 50):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in data_gen.batch_gen(x_tst, y_tst, 50):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))

