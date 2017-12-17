import datetime
import logging
import sys
import ujson
from functools import reduce

import plac

from neural_nlp.datagen.stackoverflow import StackOverflow
from neural_nlp.models.dssm import create_model
from neural_nlp.nnet.model import MetaData
from neural_nlp.nnet.timer import EpochTimer


@plac.annotations(
    identifier=('Model identifier', 'option', 'i'),
    epochs=('Epochs', 'option', 'e', int),
    batch_sz=('Batch size', 'option', 'b', int))
def main(identifier, epochs=10, batch_sz=128):
    TIMESTAMP = datetime.datetime.now()
    file_handler = logging.FileHandler(filename='logging/stackoverflow_%s.log' % TIMESTAMP.strftime('%Y%m%d_%H%M'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, handlers=handlers)

    logging.info('#' * 80)
    logging.info('Building model...')

    so = StackOverflow()
    so.initialize()
    feature_dim = len(so._cv.vocabulary_)

    model, encoder = create_model(feature_dim)


    meta_misc = {'epochs': epochs,
                 'batch_sz': batch_sz,
                 'model_json': ujson.loads(model.to_json())}

    n_records = reduce(lambda a, b: a+b[0].shape[0], so.batcher(), 0)

    meta_misc['train_records'] = n_records
    # meta_misc['test_records'] = n_test_records

    logging.info(model.metrics_names)

    meta_data = MetaData(identifier, model, TIMESTAMP, meta_misc)
    meta_data.write('models')

    logging.info('// MODEL RUN: %s //' % meta_data.run_timestamp())
    model.summary()

    epoch_time = EpochTimer(epochs, n_records, batch_sz)

    for epoch in range(epochs):
        epoch += 1
        epoch_time.set_epoch_start_time()
        logging.info('// EPOCH: %s //' % str(epoch))

        # gen_data = batch_generator(meta_misc['train'], feature_dim, batch_sz,
        #                            tfidf=True, tfidf_data=tfidf_data, occ_groups=occ_groups, multioutput=multioutput,
        #                            targets_in_1k=targets_in_1k, dropout=dropout,
        #                            exclude_skills=exclude_skills, exclude_edu=exclude_edu, exclude_exp=exclude_exp)

        for batch_num, batch in enumerate(so.batcher()):
            metrics = model.train_on_batch(batch, [[1,0,0,0]]*batch[0].shape[0])
            if batch_num % 10 == 0:
                epoch_time.hms_message(epoch, batch_num + 1, str(metrics))


        logging.info(eval)

        model_file = 'models/model_dssm_%s_e%d.h5' % (meta_data.run_timestamp(), epoch)
        logging.info('// SAVING CHECKPOINT: %s //' % model_file)
        model.save(model_file)


if __name__ == '__main__':
    plac.call(main)

