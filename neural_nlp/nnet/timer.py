import datetime
import logging


class EpochTimer:
    def __init__(self, n_epochs, n_records, batch_sz):
        self.start_time = datetime.datetime.now()
        self.epoch_start_time = datetime.datetime.now()
        self.iters_per_epoch = n_records / batch_sz
        self.n_epochs = n_epochs

    def set_epoch_start_time(self):
        self.epoch_start_time = datetime.datetime.now()

    def _hours(self, seconds):
        return seconds / 3600

    def _minutes(self, seconds):
        return seconds / 60 % 60

    def _seconds(self, seconds):
        return seconds % 60


    def hms(self, epoch_number, iter_number):
        if epoch_number < 1 or iter_number < 1:
            raise ValueError('Epochs and iterations must be 1-indexed.')
        now = datetime.datetime.now()
        amount_done = 1.0 * iter_number / self.iters_per_epoch

        td = (now - self.epoch_start_time).seconds
        epoch_time_left = (td * 1.0 / amount_done) - td
        tot_td = (now - self.start_time).seconds
        tot_amount_done = (epoch_number - 1.0 + amount_done) / self.n_epochs
        tot_time_left = (tot_td * 1.0 / tot_amount_done) - tot_td
        if epoch_number == self.n_epochs:
            tot_time_left = epoch_time_left

        return (
        '%02d:%02d:%02d' % (self._hours(tot_time_left), self._minutes(tot_time_left), self._seconds(tot_time_left)),
        '%02d:%02d:%02d' % (self._hours(epoch_time_left), self._minutes(epoch_time_left), self._seconds(epoch_time_left)))

    def hms_message(self, epoch_number, iter_number, message):
        hms_tot, hms_epoch = self.hms(epoch_number, iter_number)
        msg = 'E[%d/%d|%s] B[%d/%d|%s] %s' % (epoch_number, self.n_epochs, hms_tot, iter_number, self.iters_per_epoch,
                                              hms_epoch, message)
        logging.info(msg)



