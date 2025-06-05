### NAME: Pooja A
### Reg.No: 212222240072
### Date: 18/03/25
# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000
   data points using the ArmaProcess class.Plot the generated time series and set the title and x-axis limits.
4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
   plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000
    data points using the ArmaProcess class.Plot the generated time series and set the title and x-
   axis limits.
6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
  plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
data = pd.read_csv('/content/seattle_weather_1948-2017.csv') 
print(data.head())
```
```
precipitation = data['PRCP'].dropna()
ar1 = np.array([1, -0.5])  # AR(1) coefficient
ma1 = np.array([1, 0.5])   # MA(1) coefficient
arma11_process = ArmaProcess(ar1, ma1)
arma11_sample = arma11_process.generate_sample(nsample=len(precipitation))

ar2 = np.array([1, -0.5, 0.25])  # AR(2) coefficients
ma2 = np.array([1, 0.4, 0.3])    # MA(2) coefficients
arma22_process = ArmaProcess(ar2, ma2)

arma22_sample = arma22_process.generate_sample(nsample=len(precipitation))

plt.figure(figsize=(14, 8))

plt.subplot(221)
plot_acf(arma11_sample, lags=20, ax=plt.gca(), title='ACF of Simulated ARMA(1,1)')
plt.subplot(222)
plot_pacf(arma11_sample, lags=20, ax=plt.gca(), title='PACF of Simulated ARMA(1,1)')

plt.subplot(223)
plot_acf(arma22_sample, lags=20, ax=plt.gca(), title='ACF of Simulated ARMA(2,2)')
plt.subplot(224)
plot_pacf(arma22_sample, lags=20, ax=plt.gca(), title='PACF of Simulated ARMA(2,2)')

plt.tight_layout()
plt.show()


model = ARIMA(precipitation, order=(2,0,2))  # Using ARMA(2,2) as an example
fitted_model = model.fit()

print(fitted_model.summary())

residuals = fitted_model.resid
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals of ARMA(2,2) Model')
plt.show()
```
### OUTPUT:
#### SIMULATED ARMA(1,1) PROCESS:

Partial Autocorrelation



![image](https://github.com/user-attachments/assets/e2ad1d66-cc78-4f73-aaaf-dcc79fed3cba)



Autocorrelation


![image](https://github.com/user-attachments/assets/eba3765e-fd2f-45e4-aace-591365f98864)




##### SIMULATED ARMA(2,2) PROCESS:

Partial Autocorrelation


![image](https://github.com/user-attachments/assets/dcd1d2ea-16f6-403a-8b0e-55b16a7d6fae)



Autocorrelation


![image](https://github.com/user-attachments/assets/3f0a1dd5-dac7-44e7-b4f1-02149d56b8ff)


### RESULT:
Thus, a python program is created to fit ARMA Model successfully.
