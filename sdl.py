import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from behaviordisc import cp_detection_KSWIN, tp_detection
import re
from statsmodels.tsa.stattools import adfuller, acf
from scipy.fftpack import fft, fftfreq
from math import ceil

EXPECTED_PERIODS = {'1H': [24, 168, 672], # expected periods for seasonal patterns: Later can be set by user
                    '8H': [3, 21, 84],
                    '1D': [7, 28, 352],
                    '7D': [4, 48]}


# Test for multivariate ts
def stationary_test(data):  # df input
    """
    Tests for all variables if its stationary using adf-test.
    If at least one feature is not, returns false for the corresponding data
    :param data: df object, i.e. sd_log.data
    :return:
    """
    for feat in data.columns:
        result = adfuller(data[feat], autolag='AIC')
        if result[1] > 0.05:
            print(str(feat) + ' is not stationary')
            return False
    return True


def make_stationary(data, count=0):  # df input
    """
    Makes the data stationary using diff
    :param data: df object, i.e. sd_log.data
    :param count: counts order of differencing
    :return: stationary data as df and order of differencing
    """
    if stationary_test(data):
        return data, count
    else:
        return make_stationary(data.diff().dropna(), count + 1)


def get_period(tw, n_weeks):
    # tw one of ['1H', '8H', '1D', '7D'] TODO might be more
    if tw == '1H':
        period = n_weeks * 168
    elif tw == '8H':
        period = n_weeks * 21
    elif tw == '1D':
        period = n_weeks * 7
    elif tw == '7D':
        period = 4
    else:
        period = None

    return period


class Sdl:

    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.raw_data = pd.read_csv(path)

        self.series = self.data.to_numpy()
        self.columns = self.data.columns
        self.tw = re.findall(r'\d+[A-Z]', self.columns[0])[0]  # time window of sd_log
        #  variables as string
        self.arrival_rate = [s for s in self.columns if "arrival" in s.lower()][0]
        self.finish_rate = [s for s in self.columns if "finish" in s.lower()][0]
        self.num_unique_resource = [s for s in self.columns if "resource" in s.lower()][0]
        self.process_active_time = [s for s in self.columns if "active" in s.lower()][0]
        self.service_time = [s for s in self.columns if "service" in s.lower()][0]
        # TODO
        self.time_in_process = [self.columns[5]][0]
        self.waiting_time = [s for s in self.columns if "waiting" in s.lower()][0]
        self.num_in_process = [self.columns[7]][0]

        self.isStationary = stationary_test(self.data)
        self.period = self.estimate_period()
        self.data_diff = make_stationary(self.data)
        # TODO
        self.relations = {}
        self.changepoints = {}
        self.turningpoints = {}
        # self.calc_turning_points()
        self.behavior = {}

    def preprocess_rawData(self):
        #  TODO, currently expecting Active (preprocessed) sdLog
        data = self.rawData
        data = data.fillna(method='pad')  # filling missing values with previous ones

        return data

    # returns points as numpy array
    def get_points(self, col):
        return np.array(self.data[col])

    # plots all aspect
    def plot_all(self, title='All aspects plotted:', save=False):
        self.data.plot(subplots=True, xlabel="index",
                       figsize=(5, 10), grid=True)
        plt.show()
        if save:
            plt.savefig('pictures/' + title)

    def plot_all_with_cp(self):
        ax = self.data.plot(subplots=True, xlabel="index",
                            figsize=(5, 10), grid=True)

        for i, col in zip(ax, self.columns):
            detected = cp_detection_KSWIN(self.get_points(col), period=self.tw)
            if not detected:
                continue
            i.axvspan(0, detected[0], label="Change Point", color="red", alpha=0.3)
            for s in range(0, len(detected) - 2, 2):
                i.axvspan(detected[s], detected[s + 1], label="Change Point", color="green", alpha=0.3)
                i.axvspan(detected[s + 1], detected[s + 2], label="Change Point", color="red", alpha=0.3)
            i.axvspan(detected[-1], len(self.data), label="Change Point", color="green", alpha=0.3)
        plt.show()

    def calc_turning_points(self):
        for feat in self.columns:
            series = self.get_points(feat)
            #period = get_period(self.tw, n_weeks=1)
            tps = tp_detection(series, period=self.period)
            self.turningpoints[feat] = tps.to_numpy()

    def estimate_period(self):  # estimates period based on arrival rate in seasonal pattern
        """
        1) estimates period by computing FFT (periodogram) and find Time Periods within the Top 3 Highest Power
        2) compare period to expected ones, if they match compute acf to check significance
        """
        series = self.get_points(self.arrival_rate)
        if not self.isStationary:
            series = np.diff(series)

        # get top 3 seasons
        no_of_seasons = 3
        series_fft = fft(series)

        power = np.abs(series_fft)
        sample_freq = fftfreq(series_fft.size)

        # Find the peak frequency
        pos_mask = np.where(sample_freq > 0)
        freqs = sample_freq[pos_mask]
        powers = power[pos_mask]

        # find top frequencies and corresponding time periods for seasonal pattern
        top_powers = np.argpartition(powers, -no_of_seasons)[-no_of_seasons:]

        time_periods_from_fft = 1 / freqs[top_powers]
        time_periods = time_periods_from_fft.astype(int)
        print(time_periods)

        time_lags_expected = EXPECTED_PERIODS[self.tw]
        # One of the seasonality returned from FFT should be within range of Expected time period
        for time_lag in time_lags_expected:
            nearest_time_lag = time_periods.flat[np.abs(time_periods - time_lag).argmin()]

            # Using 5% for range comaprison
            #tmp = range(time_lag - ceil(0.05 * time_lag), time_lag + ceil(0.05 * time_lag))
            if nearest_time_lag in range(
                    time_lag - ceil(0.05 * time_lag),
                    time_lag + ceil(0.05 * time_lag)):

                # Check ACF value with lags identified from expected
                acf_score_exp = acf(series, nlags=time_lag)[-1]
                # Check ACF value with lags identified from fft
                acf_score_fft = acf(series, nlags=nearest_time_lag)[-1]

                # Check ACF is significant or not.
                if acf_score_exp >= 2 / np.sqrt(len(series)):
                    # ACF is significant and FFT identifies seasonality
                    print('Metrics is seasonal by expected period ' + str(time_lag))
                    return time_lag
                elif acf_score_fft >= 2 / np.sqrt(len(series)):
                    # ACF is significant and FFT identifies seasonality
                    print('Metrics is seasonal by recommended period ' + str(nearest_time_lag))
                    return nearest_time_lag
                else:
                    print('ACF value of expected period is not significant')
            else:
                print('Seasonality could not be identified')
                return None
