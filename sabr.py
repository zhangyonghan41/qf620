import datetime as dt
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.optimize import least_squares
import matplotlib.pylab as plt
from scipy.interpolate import interp1d

TODAY = pd.Timestamp('2020-12-01')


def SABR(F, K, T, alpha, beta, rho, nu):
    X = K
    # if K is at-the-money-forward
    if abs(F - K) < 1e-12:
        numer1 = (((1 - beta)**2)/24)*alpha*alpha/(F**(2 - 2*beta))
        numer2 = 0.25*rho*beta*nu*alpha/(F**(1 - beta))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        VolAtm = alpha*(1 + (numer1 + numer2 + numer3)*T)/(F**(1-beta))
        sabrsigma = VolAtm
    else:
        z = (nu/alpha)*((F*X)**(0.5*(1-beta)))*np.log(F/X)
        zhi = np.log((((1 - 2*rho*z + z*z)**0.5) + z - rho)/(1 - rho))
        numer1 = (((1 - beta)**2)/24)*((alpha*alpha)/((F*X)**(1 - beta)))
        numer2 = 0.25*rho*beta*nu*alpha/((F*X)**((1 - beta)/2))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        numer = alpha*(1 + (numer1 + numer2 + numer3)*T)*z
        denom1 = ((1 - beta)**2/24)*(np.log(F/X))**2
        denom2 = (((1 - beta)**4)/1920)*((np.log(F/X))**4)
        denom = ((F*X)**((1 - beta)/2))*(1 + denom1 + denom2)*zhi
        sabrsigma = numer/denom

    return sabrsigma


def sabrcalibration(x, strikes, vols, F, T, beta):
    err = 0.0
    for i, vol in enumerate(vols):
        err += (vol - SABR(F, strikes[i], T,
                           x[0], beta, x[1], x[2]))**2

    return err

def impliedVolatility(S, K, r, price, T, payoff):
    try:
        if (payoff.lower() == 'call'):
            impliedVol = brentq(lambda x: price -
                                BlackScholesLognormalCall(S, K, r, x, T),
                                1e-12, 10.0)
        elif (payoff.lower() == 'put'):
            impliedVol = brentq(lambda x: price -
                                BlackScholesLognormalPut(S, K, r, x, T),
                                1e-12, 10.0)
        else:
            raise NameError('Payoff type not recognized')
    except Exception:
        impliedVol = np.nan

    return impliedVol


def BlackScholesLognormalCall(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def BlackScholesLognormalPut(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def calibration(df, exdate, beta, S=3662.45, zero_rate_curve=None):
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

    df.dropna(inplace=True)
    call_df = df[df['payoff'] == 'call']
    put_df = df[df['payoff'] == 'put']
    strikes = put_df['strike'].values
    impliedvols = []
    for K in strikes:
        if K > S:
            impliedvols.append(call_df[call_df['strike'] == K]['vols'].values[0])
        else:
            impliedvols.append(put_df[put_df['strike'] == K]['vols'].values[0])

    # populate "df" with the dataframe containing strikes and market implied volatilities
    df = pd.DataFrame({'strike': strikes, 'impliedvol': impliedvols})

    initialGuess = [0.02, 0.2, 0.1]
    res = least_squares(lambda x: sabrcalibration(x,
                                                  df['strike'],
                                                  df['impliedvol'],
                                                  F,
                                                  T,
                                                  beta),
                        initialGuess)
    alpha = res.x[0]
    rho = res.x[1]
    nu = res.x[2]

    print('Calibrated SABR model parameters: alpha = %.3f, beta = %.1f, rho = %.3f, nu = %.3f' % (alpha, beta, rho, nu))

    sabrvols = []
    for K in strikes:
        sabrvols.append(SABR(F, K, T, alpha, beta, rho, nu))

    plt.figure(tight_layout=True)
    plt.title(f'SPX: Calibrated SABR model (exdate={exdate})')
    plt.plot(strikes, df['impliedvol'], 'gs', label='Market Vols')
    plt.plot(strikes, sabrvols, 'm--', label='SABR Vols')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Load csv files
    spx_df = pd.read_csv('SPX_options.csv')
    rates_df = pd.read_csv('zero_rates_20201201.csv')

    # mid prices
    spx_df['mid'] = 0.5 * (spx_df['best_bid'] + spx_df['best_offer'])

    # strike values
    spx_df['strike'] = spx_df['strike_price'] * 0.001

    # payoff: 'C' -> 'call', 'P' -> put
    spx_df['payoff'] = spx_df['cp_flag'].map(lambda x: 'call' if x == 'C' else 'put')

    # expiry dates
    exdates = sorted(spx_df['exdate'].unique())

    # get rates
    zero_rate_curve = interp1d(rates_df['days'], rates_df['rate'])  # 构造插值函数
    #
    for exdate in exdates:
        calibration(df=spx_df, exdate=exdate, beta=0.7,
                    zero_rate_curve=zero_rate_curve
                    # rates=rates
                    # rates=0.14/100.0
                    )
    #
    # for exdate in exdates:
    #     displaced_diffusion(
    #         df=spx_df, exdate=exdate, zero_rate_curve=zero_rate_curve)
