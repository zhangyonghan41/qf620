import array
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import datetime as dt
from scipy import interpolate
from utils import (BlackScholesPut, BachelierPut, BachelierCall, impliedPutVolatility,
                   impliedCallVolatility, BlackScholesCall,
                   calculate_atm_volatility, american_put,
                   european_put, american_call, european_call,
                   DisplacedDiffusionPut, DisplacedDiffusionCall)



def DisplacedDiffusionVolatility_y(spy, S, r, sigma, T, betas,
                                   title='Displaced-Diffusion Volatility - Ex-Date: 18/12/2020', steps=20):
    spy[(spy.cp_flag == "C")].reset_index(drop=True)

    spy_call = spy[(spy.cp_flag == "C")].reset_index(drop=True)
    spy_put = spy[(spy.cp_flag == "P")].reset_index(drop=True)
    strikes = spy_put["strike_price"].values

    summary_T = []
    for i, K in enumerate(strikes):
        if K <= S:
            AP = american_put(S, K, r, sigma, T, steps)
            EP = european_put(S, K, r, sigma, T, steps)
        else:
            AP = american_call(S, K, r, sigma, T, steps)
            EP = european_call(S, K, r, sigma, T, steps)

        summary_T.append([K, AP, EP])

    T_df = pd.DataFrame(summary_T, columns=['strike', 'AP', 'EP'])
    T_df['Premium'] = T_df['AP'] - T_df["EP"]

    spy_put['mid_price_stripped'] = spy_put['mid_price'] - T_df['Premium']
    spy_call['mid_price_stripped'] = spy_call['mid_price'] - T_df['Premium']

    summary = []
    for i, K in enumerate(strikes):
        to_append = [K, ]

        if K <= S:
            impliedvol_market = impliedPutVolatility(S, K, r, spy_put['mid_price_stripped'][i], T)
            price_lognormal = BlackScholesPut(S, K, r, sigma, T)
            impliedvol_lognormal = impliedPutVolatility(S, K, r, price_lognormal, T)

            price_normal = BachelierPut(S, K, r, sigma, T)
            impliedvol_normal = impliedPutVolatility(S, K, r, price_normal, T)

            to_append += [
                impliedvol_market,
                impliedvol_lognormal,
                impliedvol_normal]

            for beta in betas:
                price_dd = DisplacedDiffusionPut(S, K, r, sigma, T, beta)
                impliedvol_dd = impliedPutVolatility(S, K, r, price_dd, T)
                to_append.append(impliedvol_dd)

        else:
            impliedvol_market = impliedCallVolatility(S, K, r, spy_call['mid_price_stripped'][i], T)

            price_lognormal = BlackScholesCall(S, K, r, sigma, T)
            impliedvol_lognormal = impliedCallVolatility(S, K, r, price_lognormal, T)

            price_normal = BachelierCall(S, K, r, sigma, T)
            impliedvol_normal = impliedCallVolatility(S, K, r, price_normal, T)

            to_append += [
                impliedvol_market,
                impliedvol_lognormal,
                impliedvol_normal]

            for beta in betas:
                price_dd = DisplacedDiffusionCall(S, K, r, sigma, T, beta)
                impliedvol_dd = impliedCallVolatility(S, K, r, price_dd, T)
                to_append.append(impliedvol_dd)

        summary.append(to_append)

    columns = ['strike', 'impliedvol_market', 'impliedvol_lognormal', 'impliedvol_normal',]
    columns += [f'impliedvol_dd{i+1}' for i in range(len(betas))]
    dd = pd.DataFrame(summary, columns=columns)
    plt.plot(dd['strike'], dd['impliedvol_market'], 'gs', label='Market Volatility')

    for i, beta in enumerate(betas):
        sns.lineplot(x=dd['strike'], y=dd[f'impliedvol_dd{i+1}'], label=f'Beta = {beta}',
                     linestyle='--')
    sns.lineplot(x=dd['strike'], y=dd['impliedvol_normal'], label='Beta = 0 (normal model)', linestyle='--')
    sns.lineplot(x=dd['strike'], y=dd['impliedvol_lognormal'], label='Beta = 1 (lognormal model)', linestyle='--')

    plt.legend()
    plt.title(title, fontsize=10)
    plt.ylabel('Volatility')
    plt.show()


if __name__ == '__main__':
    today = dt.date(2020, 12, 1)
    # Discount Rate
    rate_df = pd.read_csv('zero_rates_20201201.csv')

    # SPY General Data
    spy_df = pd.read_csv('SPY_options.csv')
    spy_df['strike_price'] = spy_df['strike_price'] / 1000
    spy_df['mid_price'] = (spy_df['best_bid'] + spy_df['best_offer']) / 2

    exdates = np.unique(spy_df.exdate)

    for exdate in exdates:
        spy = spy_df[(spy_df.exdate == exdate)]
        exdate = pd.Timestamp(str(exdate)).date()
        T = (exdate - today).days / 365.0
        # Discount Rate Interpolation
        x = rate_df['days']
        y = rate_df['rate']
        f = interpolate.interp1d(x, y)
        r = f(T * 365) / 100

        # Underlying Value & ATM Strike Price
        S = 366.02
        K = 366

        # Beta Parameter (Displaced-Diffusion Model)
        beta1 = 0.8
        beta2 = 0.6
        beta3 = 0.4
        beta4 = 0.2
        betas = [beta1, beta2, beta3, beta4]

        sigma = calculate_atm_volatility(spy, S, K, r, T)

        DisplacedDiffusionVolatility_y(
            spy, S, r, sigma, T, betas,
            title=f'SPY-DisplacedDiffusion-Sigma{sigma}-ExDate{exdate}')
