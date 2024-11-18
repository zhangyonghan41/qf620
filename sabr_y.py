import datetime as dt
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.optimize import minimize
from scipy.optimize import least_squares
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
import array
from sabr import impliedVolatility, sabrcalibration, SABR
from utils import (american_put,  european_put,
                   american_call, european_call)


TODAY = pd.Timestamp('2020-12-01')


def process_american_options(spy, S_A, r, sigma, T, steps):
    spy[(spy.cp_flag == "C")].reset_index(drop=True)

    spy_call = spy[(spy.cp_flag == "C")].reset_index(drop=True)
    spy_put = spy[(spy.cp_flag == "P")].reset_index(drop=True)
    strikes = spy_put["strike"].values

    # spy_call = spy[(spy.cp_flag == "C")].reset_index(drop=True)
    # spy_put = spy[(spy.cp_flag == "P")].reset_index(drop=True)
    # strikes = spy_put["strike_price"].values

    summary_T = []
    for i, K in enumerate(strikes):
        if K <= S_A:
            AP = american_put(S_A, K, r, sigma, T, steps)
            EP = european_put(S_A, K, r, sigma, T, steps)
        else:
            AP = american_call(S_A, K, r, sigma, T, steps)
            EP = european_call(S_A, K, r, sigma, T, steps)

        summary_T.append([K, AP, EP])

    T_df = pd.DataFrame(summary_T, columns=['strike', 'AP', 'EP'])
    T_df['Premium'] = T_df['AP'] - T_df["EP"]

    spy_put['mid_price_stripped'] = spy_put['mid'] - T_df['Premium']
    spy_call['mid_price_stripped'] = spy_call['mid'] - T_df['Premium']

    summary = []
    for i, K in enumerate(strikes):
        if K <= S_A:
            impliedvol_market = impliedVolatility(S=S_A, K=K, r=r,
                                                  price=spy_put['mid_price_stripped'][i],
                                                  T=T, payoff='put')
        else:
            impliedvol_market = impliedVolatility(S=S_A, K=K, r=r,
                                                  price=spy_call['mid_price_stripped'][i],
                                                  T=T, payoff='call')
        summary.append([K, impliedvol_market])

    return pd.DataFrame(summary, columns=['strike', 'impliedvol'])


# For american options
def calibration_y(df, exdate, beta, steps=20, S=3662.45, zero_rate_curve=None):
    df = df[df['exdate'] == exdate]
    days_to_expiry = (pd.Timestamp(str(exdate)) - TODAY).days
    T = days_to_expiry / 365
    r = zero_rate_curve(days_to_expiry) / 100  # interpolation
    F = S * np.exp(r * T)

    df['vols'] = df.apply(lambda x: impliedVolatility(S,
                                                      x['strike'],
                                                      r,
                                                      x['mid'],
                                                      T,
                                                      x['payoff']),
                          axis=1)
    sigma = np.interp(S, df["strike"], df['vols'])

    df = process_american_options(spy=df, S_A=S, r=r, sigma=sigma, T=T, steps=steps)
    # df.dropna(inplace=True)

    initialGuess = [0.1, 0.1, 0.1]
    res = least_squares(lambda x: sabrcalibration(x,
                                                  df['strike'].values,
                                                  df['impliedvol'].values,
                                                  F,
                                                  T,
                                                  beta),
                        initialGuess)
    alpha = res.x[0]
    rho = res.x[1]
    nu = res.x[2]

    print('Calibrated SABR model parameters: alpha = %.3f, beta = %.1f, rho = %.3f, nu = %.3f' % (alpha, beta, rho, nu))

    sabrvols = []
    for K in df['strike'].values:
        sabrvols.append(SABR(F, K, T, alpha, beta, rho, nu))

    plt.figure(tight_layout=True)
    plt.title(f'SPY: Calibrated SABR model (exdate={exdate})')
    plt.plot(df['strike'].values, df['impliedvol'], 'gs', label='Market Vols')
    plt.plot(df['strike'].values, sabrvols, 'm--', label='SABR Vols')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Load csv files
    spy_df = pd.read_csv('SPY_options.csv')
    rates_df = pd.read_csv('zero_rates_20201201.csv')

    # mid prices
    spy_df['mid'] = 0.5 * (spy_df['best_bid'] + spy_df['best_offer'])

    # strike values
    spy_df['strike'] = spy_df['strike_price'] * 0.001

    # payoff: 'C' -> 'call', 'P' -> put
    spy_df['payoff'] = spy_df['cp_flag'].map(lambda x: 'call' if x == 'C' else 'put')

    # expiry dates
    exdates = sorted(spy_df['exdate'].unique())

    # get rates
    zero_rate_curve = interp1d(rates_df['days'], rates_df['rate'])  # 构造插值函数
    #
    for exdate in exdates:
        calibration_y(df=spy_df, exdate=exdate, beta=0.7,
                    zero_rate_curve=zero_rate_curve,
                    S=366.02  # 3662.45
                    # rates=rates
                    # rates=0.14/100.0
                    )
    #
    # for exdate in exdates:
    #     displaced_diffusion(
    #         df=spx_df, exdate=exdate, zero_rate_curve=zero_rate_curve)
