import numpy as np
import array
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
import matplotlib.pylab as plt
import seaborn as sns
from scipy.optimize import least_squares


def calculate_atm_volatility(spx_data, S, K, r, T):
    atm_call = spx_data[(spx_data['strike_price'] == K) & (spx_data['cp_flag'] == 'C')]
    atm_put = spx_data[(spx_data['strike_price'] == K) & (spx_data['cp_flag'] == 'P')]

    if not atm_call.empty and not atm_put.empty:
        sigma_call = impliedVolatility(S, K, r, atm_call.iloc[0]['mid_price'], T, 'C')
        sigma_put = impliedVolatility(S, K, r, atm_put.iloc[0]['mid_price'], T, 'P')
        return (sigma_call + sigma_put) / 2
    else:
        return np.nan


# Displaced-Diffusion Model
def DisplacedDiffusionCall(S, K, r, sigma, T, beta):
    F = np.exp(r*T)*S
    c1 = (np.log(F/(F+beta*(K-F)))+(beta*sigma)**2/2*T) / (beta*sigma*np.sqrt(T))
    c2 = c1 - beta*sigma*np.sqrt(T)
    return np.exp(-r*T)*(F/beta*norm.cdf(c1) - ((1-beta)/beta*F + K)*norm.cdf(c2))

def DisplacedDiffusionPut(S, K, r, sigma, T, beta):
    F = np.exp(r*T)*S
    c1 = (np.log(F/(F+beta*(K-F)))+(beta*sigma)**2/2*T) / (beta*sigma*np.sqrt(T))
    c2 = c1 - beta*sigma*np.sqrt(T)
    return np.exp(-r*T)*(((1-beta)/beta*F + K)*norm.cdf(-c2) - F/beta*norm.cdf(-c1))


# Implied European Options Volatility Model
def impliedCallVolatility(S, K, r, price, T):
    try:
        impliedVol = brentq(lambda x: price -
                        BlackScholesCall(S, K, r, x, T),
                        1e-6, 10)
    except Exception:
        impliedVol = np.nan
    return impliedVol



# Bachelier Model
def BachelierCall(S, K, r, sigma, T):
    d = (S-K) / (S*sigma*np.sqrt(T))
    disc = np.exp(-r*T)
    return disc*((S-K)*norm.cdf(d)+S*sigma*np.sqrt(T)*norm.pdf(d))

def BachelierPut(S, K, r, sigma, T):
    d = (S-K) / (S*sigma*np.sqrt(T))
    disc = np.exp(-r*T)
    return disc*((K-S)*norm.cdf(-d)+S*sigma*np.sqrt(T)*norm.pdf(-d))


def BlackScholesPut(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

# Black-Scholes Model
def BlackScholesCall(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def impliedPutVolatility(S, K, r, price, T):
    try:
        impliedVol = brentq(lambda x: price -
                        BlackScholesPut(S, K, r, x, T),
                        1e-6, 10)
    except Exception:
        impliedVol = np.nan
    return impliedVol

def impliedVolatility(S, K, r, price, T, option_type):
    try:
        if option_type == 'C':
            return brentq(lambda x: price - BlackScholesCall(S, K, r, x, T), 1e-12, 10)
        else:
            return brentq(lambda x: price - BlackScholesPut(S, K, r, x, T), 1e-12, 10)
    except Exception:
        return np.nan

def american_put(S, K, r, sigma, T, steps):
    R = (1+r)**(T/steps)
    u = np.exp(sigma*np.sqrt(T/steps))
    uu = u*u
    d = 1.0/u
    p_up = (R-d)/(u-d)
    p_down = 1.0-p_up
    prices = array.array('d', (0 for i in range(0, steps+1)))
    prices[0] = S*pow(d, steps)

    for i in range(1, steps+1):
        prices[i] = uu*prices[i-1]

    put_values = array.array('d', (0 for i in range(0, steps+1)))

    for i in range(0, steps+1):
        put_values[i] = max(0.0, (K-prices[i]))

    for step in range(steps-1, -1, -1):
        for i in range(0, step+1):
            put_values[i] = (p_up*put_values[i+1]+p_down*put_values[i])/R
            prices[i] = d*prices[i+1]
            put_values[i] = max(put_values[i], (K-prices[i]))
    return put_values[0]

def european_put(S, K, r, sigma, T, steps):
    R = (1+r)**(T/steps)
    u = np.exp(sigma*np.sqrt(T/steps))
    uu = u*u
    d = 1.0/u
    p_up = (R-d)/(u-d)
    p_down = 1.0-p_up
    prices = array.array('d', (0 for i in range(0, steps+1)))
    prices[0] = S*pow(d, steps)

    for i in range(1, steps+1):
        prices[i] = uu*prices[i-1]

    put_values = array.array('d', (0 for i in range(0, steps+1)))

    for i in range(0, steps+1):
        put_values[i] = max(0.0, (K-prices[i]))

    for step in range(steps-1, -1, -1):
        for i in range(0, step+1):
            put_values[i] = (p_up*put_values[i+1]+p_down*put_values[i])/R
            prices[i] = d*prices[i+1]
    return put_values[0]

def american_call(S, K, r, sigma, T, steps):
    R = (1+r)**(T/steps)
    u = np.exp(sigma*np.sqrt(T/steps))
    d = 1.0/u
    p_up = (R-d)/(u-d)
    p_down = 1.0-p_up

    prices = array.array('d', (0 for i in range(0, steps+1)))

    prices[0] = S*pow(d, steps)
    uu = u*u
    for i in range(1, steps+1):
        prices[i] = uu*prices[i-1]

    call_values = array.array('d', (0 for i in range(0, steps+1)))
    for i in range(0, steps+1):
        call_values[i] = max(0.0, (prices[i]-K))

    for step in range(steps-1, -1, -1):
        for i in range(0, step+1):
            call_values[i] = (p_up*call_values[i+1]+p_down*call_values[i])/R
            prices[i] = d*prices[i+1]
            call_values[i] = max(call_values[i], 1.0*(prices[i]-K))

    return call_values[0]

def european_call(S, K, r, sigma, T, steps):
    R = (1+r)**(T/steps)
    u = np.exp(sigma*np.sqrt(T/steps))
    d = 1.0/u
    p_up = (R-d)/(u-d)
    p_down = 1.0-p_up

    # price of underlying
    prices = array.array('d', (0 for i in range(0, steps+1)))

    # fill in the endnodes
    prices[0] = S*pow(d, steps)
    uu = u*u
    for i in range(1, steps+1):
        prices[i] = uu*prices[i-1]

    call_values = array.array('d', (0 for i in range(0, steps+1)))
    for i in range(0, steps+1):
        call_values[i] = max(0.0, (prices[i]-K))

    for step in range(steps-1, -1, -1):
        for i in range(0, step+1):
            call_values[i] = (p_up*call_values[i+1]+p_down*call_values[i])/R
            prices[i] = d*prices[i+1]

    return call_values[0]


def process_american_options(spy, S_A, r, sigma, T, steps):
    spy_call = spy[(spy.cp_flag == "C")].reset_index(drop=True)
    spy_put = spy[(spy.cp_flag == "P")].reset_index(drop=True)
    strikes = spy_put["strike_price"].values

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

    spy_put['mid_price_stripped'] = spy_put['mid_price'] - T_df['Premium']
    spy_call['mid_price_stripped'] = spy_call['mid_price'] - T_df['Premium']

    summary = []
    for i, K in enumerate(strikes):
        if K <= S_A:
            impliedvol_market = impliedPutVolatility(S_A, K, r, spy_put['mid_price_stripped'][i], T)
        else:
            impliedvol_market = impliedCallVolatility(S_A, K, r, spy_call['mid_price_stripped'][i], T)

        summary.append([K, impliedvol_market])

    return pd.DataFrame(summary, columns=['strike', 'impliedvol_market'])


# SABR Calibration
def sabrcalibration(x, strikes, vols, F, T, beta):
    err = 0.0
    for i, vol in enumerate(vols):
        err += (vol - SABR(F, strikes[i], T,
                           x[0], beta, x[1], x[2]))**2
    return err


# SABR Model
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

def sabr_volatility_calib_plot(dd, r, T, S, beta, exdate):
    F = np.exp(r * T) * S
    initialGuess = [0.1, 0.1, 0.1]
    res = least_squares(lambda x: sabrcalibration(x, dd['strike'], dd['impliedvol_market'], F, T),
                        initialGuess)

    alpha = res.x[0]
    rho = res.x[1]
    nu = res.x[2]

    SABR_values = [SABR(F, K, T, alpha, beta, rho, nu) for K in dd['strike']]

    print(f'SABR Calibration for {exdate}: alpha = {alpha:.3f}, beta = {beta:.1f}, rho = {rho:.3f}, nu = {nu:.3f}')

    plt.figure(figsize=(18, 6))
    plt.plot(dd['strike'], dd['impliedvol_market'], 'k--', label='Market (Implied Vol)')
    sns.lineplot(x=dd['strike'], y=SABR_values, label='SABR Vol', linestyle='--')
    plt.title(f'SABR Volatility - ExDate: {exdate}', fontsize=20)
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.legend()
    plt.show()
