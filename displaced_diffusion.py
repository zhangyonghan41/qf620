import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import datetime as dt
from scipy import interpolate
from utils import (BlackScholesPut, BachelierPut, BachelierCall, impliedPutVolatility,
                   impliedCallVolatility, BlackScholesCall,
                   calculate_atm_volatility,
                   DisplacedDiffusionPut, DisplacedDiffusionCall)


def DisplacedDiffusionVolatility(spx, S, r, sigma, T, betas,
                                 title='Displaced-Diffusion Volatility - Ex-Date: 18/12/2020'):
    spx_call = spx[(spx.cp_flag == "C")].reset_index(drop=True)
    spx_put = spx[(spx.cp_flag == "P")].reset_index(drop=True)
    strike = spx_put["strike_price"].values

    summary = []
    for i in range(len(spx_put.index)):
        K = strike[i]

        to_append = [K, ]

        if K <= S:
            impliedvol_market = impliedPutVolatility(S, K, r, spx_put['mid_price'][i], T)

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

        elif K > S:
            impliedvol_market = impliedCallVolatility(S, K, r, spx_call['mid_price'][i], T)

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

    # SPX General Data
    spx_df = pd.read_csv('SPX_options.csv')
    spx_df['strike_price'] = spx_df['strike_price'] / 1000
    spx_df['mid_price'] = (spx_df['best_bid'] + spx_df['best_offer']) / 2

    exdates = np.unique(spx_df.exdate)

    for exdate in exdates:
        spx = spx_df[(spx_df.exdate == exdate)]
        exdate = pd.Timestamp(str(exdate)).date()
        T = (exdate - today).days / 365.0
        # Discount Rate Interpolation
        x = rate_df['days']
        y = rate_df['rate']
        f = interpolate.interp1d(x, y)
        r = f(T * 365) / 100

        # Underlying Value & ATM Strike Price
        S = 3662.45
        K = 3660

        # Beta Parameter (Displaced-Diffusion Model)
        beta1 = 0.8
        beta2 = 0.6
        beta3 = 0.4
        beta4 = 0.2
        betas = [beta1, beta2, beta3, beta4]

        # Steps Parameter (American Options)
        steps = 20

        sigma = calculate_atm_volatility(spx, S, K, r, T)

        DisplacedDiffusionVolatility(
            spx, S, r, sigma, T, betas,
            title=f'Displaced-Diffusion Volatility - Ex-Date: {exdate}')
