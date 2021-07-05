"""
- univariate forecasting using SARIMA in auto-arima
- multivariate forecasting using var
- some diagnostic plots
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from pmdarima.arima.utils import ndiffs
from pmdarima.utils import diff_inv


def stats_plot_acf(series):
    plot_acf(series)
    plt.show()


def stats_plot_pacf(series):
    plot_pacf(series)
    plt.show()


def arima_model(series):
    # 1,1,2 ARIMA Model
    model = ARIMA(series, order=(3, 0, 3))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    model_fit.plot_predict(dynamic=False)
    plt.show()


def auto_arima(series, freq):
    """
    Builds SARIMA model used in univariate forecasting
    :param series: numy array or pandas series
    :param freq: periodicity of sd_log. Use period in SdLog Obejct
    :return: model
    """
    # if tw == '1H':
    #     freq = 168
    # elif tw == '8H':
    #     freq = 21
    # elif tw == '1D':
    #     freq = 7
    # elif tw == '7D':
    #     freq = 4
    # else:
    #     freq = 1
    if not freq:
        freq = 1
    model = pm.auto_arima(series, start_p=1, start_q=1,
                          test='adf',  # use adftest to find optimal 'd'
                          max_p=3, max_q=3,  # maximum p and q
                          m=freq,  # frequency of series
                          d=None,  # let model determine 'd'
                          seasonal=True,  # Seasonality
                          start_P=0,
                          D=None,  # let model determine 'D'
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)
    model.summary()
    return model


def uni_forecast(series, n_periods, freq, save_plot=False, outputpath=None):
    """
    This method used the auto_arima model to forecast univariate time series using the SARIMA model.
    The auto_model iterates over different combinations of the parameters and uses the AIC to find the optimal one
    :param series: numpy array or pandas series of univariate time series
    :param n_periods: number of steps we would like to forecast
    :param freq: periodicity of sd_log. Use period in SdLog Obejct
    :return: predicted values as df with corresponding index
    @param save_plot:
    @param outputpath:
    """
    # Forecast
    model = auto_arima(series, freq)
    n_periods = n_periods
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = np.arange(len(series), len(series) + n_periods)

    # make series for plotting purpose
    fc_series = pd.Series(fc, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.plot(series)
    plt.plot(fc_series, color='darkgreen')
    plt.fill_between(lower_series.index,
                     lower_series,
                     upper_series,
                     color='k', alpha=.15)

    plt.title("Forecast of " + str(n_periods) + " periods")
    if save_plot:
        plt.savefig(outputpath)
    plt.show()
    return fc_series, model


def arima_diagnostic(model):
    """
    Some diagnostic to evaluate (S)ARIMA model
    :param model: arima model
    :return:
    """
    model.plot_diagnostics(figsize=(7, 5))
    plt.show()


def multi_forecast(sd_log, variables, n_period, save_plot=False, outputpath=None):  # sd_log object, variables list of features column names
    """

    :param sd_log: sd_log object
    :param variables: features you would like to use for the multivariate forecast as list
    :param n_period: steps you would like to predict
    :return:
    @param outputpath:
    @param save_plot:
    """
    max_lag = 6
    # Check for stationary
    if sd_log.isStationary:
        data = sd_log.data[variables]
    else:
        data = sd_log.data_diff[0][variables]
        ndiff = sd_log.data_diff[1]

    #  Split into train (0.9) and test (0.1)
    # data_train = data[:int(0.9*(len(data)))]
    # data_test = data[int(0.9*(len(data))):]
    model = VAR(data)
    # Look for minimum AIC/BIC and corresponding lag to fit model
    lag = min(model.select_order(maxlags=max_lag).selected_orders.values())
    lag = 1

    results = model.fit(lag)
    print(results.summary())
    var_diagnostic(results)
    results.plot_forecast(n_period)
    lag_order = results.k_ar
    fc = results.forecast(data.values[-lag_order:], n_period)
    df_fc = pd.DataFrame(fc, index=data.index[-n_period:])
    # inverting resulting forecast
    # inv_diff(sd_log.data[sd_log.finish_rate], data[sd_log.finish_rate], ndiff)
    if save_plot:
        plt.savefig(outputpath)
    plt.show()
    return df_fc


def var_diagnostic(model_fitted):
    """
    Some diagnostic to evaluate vector auto regression model
    :param model_fitted: fitted VAR model
    :return:
    """

    fevd = model_fitted.fevd(5)
    print(fevd.summary())

    ser_cor = check_ser_corr(model_fitted)
    print('Check for serial correlation (values aroun 2 are good): ' + str(ser_cor))

    model_fitted.plot_acorr()


def check_ser_corr(model_fitted):
    """
    Checking for serial correlation is to ensure that the model is sufficiently
    able to explain the variances and patterns in the time series. The value of
    this statistic can vary between 0 and 4. The closer it is to the value 2, then
    there is no significant serial correlation. The closer to 0, there is a positive
    serial correlation, and the closer it is to 4 implies negative serial correlation.
    :param model_fitted: model.fit object
    :return:
    """
    from statsmodels.stats.stattools import durbin_watson
    out = durbin_watson(model_fitted.resid)
    print(out)
    return out


def inv_diff(df_orig_column, df_diff_column, n_diff):
    if n_diff == 0:
        return df_diff_column

    x = df_orig_column.tolist()
    x_diff = df_diff_column.tolist()
    if n_diff == 1:
        x_0 = x[0]
    elif n_diff == 2:
        x_0 = x[1] - x[0]
    elif n_diff > 2:
        raise Exception('Re-transformation: only support second order differencing and lower')
    inv_values = np.r_[x_0, x_diff].cumsum()
    inv_series = pd.Series(inv_values)

    if n_diff == 1:
        return inv_series
    else:
        n_diff = n_diff - 1
        return inv_diff(df_orig_column, inv_series, n_diff)


def test(series):
    n = round(len(series) * 0.8)
    m = round(len(series) - n)
    train = series[:n]
    test_x = series[n:]

    model = auto_arima(train)
    # Forecast
    fc, conf = model.predict(n_periods=m, return_conf_int=True)
    index_of_fc = np.arange(n, len(series))

    # Make as pandas series
    fc_series = pd.Series(fc, index=index_of_fc)
    test_series = pd.Series(test_x, index=index_of_fc)
    lower_series = pd.Series(conf[:, 0], index=index_of_fc)
    upper_series = pd.Series(conf[:, 1], index=index_of_fc)

    # Calc RMSE
    rmse = np.mean((fc - test_x) ** 2) ** .5

    # Plot
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train, label='training')
    plt.plot(test_series, label='actual')
    plt.plot(fc_series, label='forecast')
    plt.fill_between(lower_series.index, lower_series, upper_series,
                     color='k', alpha=.15)
    plt.title('Forecast vs Actuals' + 'RMSE: ' + str(rmse))
    plt.legend(loc='upper left', fontsize=8)
    plt.show()


def test2(series):
    #  not auto arima
    n = round(len(series) * 0.8)
    m = round(len(series) - n)
    train = series[:n]
    test_x = series[n:]

    model = ARIMA(train, order=(3, 0, 3))
    model_fit = model.fit(disp=0)
    # Forecast
    fc, se, conf = model_fit.forecast(m, alpha=0.05)  # 95% conf
    index_of_fc = np.arange(n, len(series))

    # Make as pandas series
    fc_series = pd.Series(fc, index=index_of_fc)
    test_series = pd.Series(test_x, index=index_of_fc)
    lower_series = pd.Series(conf[:, 0], index=index_of_fc)
    upper_series = pd.Series(conf[:, 1], index=index_of_fc)

    # Calc RMSE
    rmse = np.mean((fc - test_x) ** 2) ** .5
    model_fit.plot_predict(dynamic=False)
    # Plot
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train, label='training')
    plt.plot(test_series, label='actual')
    plt.plot(fc_series, label='forecast')
    plt.fill_between(lower_series.index, lower_series, upper_series,
                     color='k', alpha=.15)
    plt.title('ARIMA(3,0,3) Forecast vs Actuals ' + 'RMSE: ' + str(rmse))
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
