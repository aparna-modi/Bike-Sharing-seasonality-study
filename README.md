# OJT Project Design Template | Bike Sharing Seasonality Study

Student Name(s): Padmanava and Aparna

Roll No(s): 251810700240 (Padmanava Parui) and 251810700036 (Aparna Modi)

Year & Section: 1st year SEM 2A - Padmanava Parui and 1st year SEM 2B - Aparna Modi

Project Title (as assigned): Bike Sharing Seasonality & Demand Analysis

Project Type: Data Science

Stack / Framework: Python, Pandas, NumPy, Sktime, YDataProfiling, MLFlow, Statsmodels, Prophet, Streamlit, Plotly, Great Expectations, MLflow  
  
Dataset: We look forward to using the UCI Bike Sharing Dataset (Hourly/Daily records) - [https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

---

  
  
  

# 1. Problem Understanding

1.1 What is the problem statement in your own words? 

Bike-sharing demand is not random; it follows "nested" cycles (daily, weekly, yearly) often obscured by noise and long-term trends. We must mathematically decompose a single time series (daily rentals) into four distinct components: Trend, Seasonal, Cyclical, and Residual.

  

1.2 Why does this problem exist or matter? 

City mobility teams need to isolate the "Seasonal" component to distinguish between organic growth and recurring patterns. This allows for accurate capacity planning (e.g., scheduling fleet maintenance in winter without affecting service) and targeted marketing strategies.

  

1.3 Key inputs and expected outputs:

|   |   |   |
|---|---|---|
|Inputs|Process|Expected Outputs|
|Historical Ride Data: Long-duration daily and hourly trip counts (Target cnt).|Trend Decomposition (LOESS): Smoothing the time series to identify underlying direction.|Trend: The long-term progression (e.g., is the program growing year-over-year?).|
|Temporal Features: Hourly timestamps, day of week, month.|Seasonal Extraction: Isolating fixed-period patterns using STL.|Seasonal: Recurring cycles (Daily: rush hour; Weekly: weekends vs. weekdays; Yearly: Summer vs. Winter).|
|External Economic Factors: Multi-year dataset covering economic shifts.|Cycle Identification: Detecting non-fixed long-term oscillations.|Cyclical: Long-term oscillations not of a fixed period (usually economic).|
|Exogenous Variables: Weather (temp, humidity) and Events (holidays, strikes).|Residual Calculation: Subtracting Trend, Season, and Cycle from Actuals (Actual - Fit).|Residual/Irregular: The "noise" or "shocks" (e.g., a sudden storm or a transit strike).|

---

  
  
  
  
  

# 2. Functional Scope

2.1 What are the core features you plan to build (must-haves)? 

  

|   |   |
|---|---|
|Feature|Description|
|Ingestion & Validation|A script (ingestion.py) to fetch data, enforce schema types, and interpolate missing timestamps to maintain time-series continuity.|
|Decomposition Engine|Implementation of STL or Classical Decomposition to mathematically separate seasonality from trend and noise.|
|Stationarity Testing|Integration of the Augmented Dickey-Fuller (ADF) test to ensure forecasts aren't biased by changing means or variances.|
|Baseline Suite|A benchmarking system comparing advanced models against a Naive Forecast (Tomorrow = Today) and a Moving Average (7-day window).|
|Backtesting|Time-Series Cross-Validation: Testing the model on a rolling "hidden" future window (e.g., Train on Months 1-12, Test on 13) to prevent data leakage.|
|Interactive Dashboard|A Streamlit app allowing users to visualize "Seasonal Demand" vs. "Capacity" with segment drill-downs (Registered vs. Casual).|

  
  

2.2 What stretch goals could you attempt if time permits? 

- Automated Alerting: Triggering a "Low Demand Alert" if Actual < (Forecast - 2*StdDev).
    
- Experiment Tracking: Using MLflow to log and compare model settings (e.g., smoothing factors).
    
- Orchestration: Using Prefect or Mage.ai to automate the pipeline (Ingest → Clean → Forecast).
    

2.3 Which libraries or tools will you use? 

- Data Engineering: Pandas, NumPy, Great Expectations (for automated quality checks).
    
- Analytical Core: Statsmodels (STL), Prophet (for holiday effects), Sktime.
    
- Visualization: Streamlit, Plotly.
    

---

  
  

# 3. System & Design Thinking

3.1 Sketch or describe your app flow / pipeline: 

We are implementing a Modular Pipeline orchestrated to ensure reproducibility. The "Validation" step acts as a gatekeeper; if the data fails the contract, the pipeline stops immediately.

The Pipeline Flow:

Ingestion (R1) → Validation (Great Expectations) [R1] → Feature Engineering → Stationarity Check (R3) → Decomposition (R2) → Baseline & Backtesting (R4/R5) → Dashboard (R6)

3.2 What data structures or algorithms are central to this project?

- STL (Seasonal-Trend decomposition using LOESS) [Satisfies R2]: The core algorithm chosen over classical decomposition because it handles outliers effectively and allows seasonal components to evolve over time (e.g., winter riding becoming more popular).
    
- Augmented Dickey-Fuller Test (ADF) [Satisfies R3]: A statistical algorithm used to check for unit roots. If the series is non-stationary (p-value > 0.05), we must apply differencing before modeling to avoid spurious results.
    
- Sliding Window Validation [Satisfies R5]: A time-series specific cross-validation structure (Train: Months 1-6, Test: Month 7) used to prevent the "data leakage" that occurs with random train-test splits.
    
- Linear Interpolation: A critical data cleaning algorithm to fill missing hourly timestamps, ensuring the time-series mathematical continuity remains unbroken.
    

3.3 How will you test correctness or performance?

- Automated Quality Checks [Validates R1]: Using Great Expectations, the pipeline will automatically fail if data violates defined rules (e.g., cnt must be positive, no missing hours), ensuring bad data never reaches the model .
    
- Business Metric (MAPE) [Validates R5]: We will use Mean Absolute Percentage Error (MAPE) instead of R², as it translates error into business terms (e.g., "We are off by 5% on average") that stakeholders can act upon.
    
- Residual "White Noise" Test [Validates R2]: A statistical proof that the residuals (errors) are random. If patterns exist in the noise, it confirms the model is missing a key driver.
    

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAV4AAABECAYAAADJEV1TAAAQAElEQVR4AeydB3xUVfbHz1BCsoskELJKKIFEJK7oEgwBElxBUcSyVAtCKIorWAANQYVgFxBll6YLAktTdBUC7l9UpBiBBCSE8PdPkaqAgEo0EZAiCv/3veGNk8nMJJmZTGaSy4f7yq3n/l7mvHPPOfe8auHh9S/opDHQfwP6b0D/Dfjub6Ca6H8aAY2ARkAj4FMENOP1Kdx6MI2ARsBvEfAhYZrx+hBsPZRGQCOgEQABzXhBQSeNgEZAI+BDBDTj9SHYeqiyI5CQkCAkWl511VXSpUsXufTSS7nVqUogUDknqRlv5XyulWJWDz74dxk+fJhMmDBe5s6dK489NkLuvvsudW0y40oxUT2JKoeAZrxV7pEHzoTj4uJk7dq1YrFUk6NHj8jgwQ8Y9+skOLiW/OlPfwqciWhKNQJ2CGjGaweIvnWMwHXXXSedO98of/zjHx1XMHKjo6MFZmlceuX/tGnTpWbNIDl//jfJyMhQfTZt2lR+/fVX+f7779W9J4e8vGOeNK8SbUeNGiVJSUn2c9X3HiKgGa+HAFb25jDaf/3rdXnyySdkwIABapkfHx9fbNrXXHONjBv3knTr1q1YmbsZ27dvl+bNL5dTp07Jrl27FdOPjW0hhw8floKCfIHRu9u3p+1uv/12mTXrDeGF5Glf/t5eM17vPyHNeL2PqU96hNG9+uqrTn/4lL/88gR59913ZeHCBZKcnOyQrsTE9jJ9+nRVb86c2QJDsa0Is8WotXDhmzJkyFD5+eefZerUKTJt2lS58847pXv3bvLSSy/KlClTpFq1aoIu1ra9J9cw/SZNmsj+/fvlwIEDcuONNxiGtctk8+YcGTZsuLRo0cKT7t1uC129evUScAkPr+d2P+XRcOzYNMnN3SJbt+a6TNnZm+Shhx4qDxJ0n6VAQDPeUoDkL1U6duyopMr09CUye/Ysad++nYSGhhYjr0OHJJk48WVDD3qpwJwzMzPlgQcGywsvPF+kLoxz3Lhxcu7cL/Lcc8/J119/LU88MUoeeeQRa72rr24pv/zyiyqD6W7b9n/yxRdfSE7OFmndOs6gIVHVPXXqZ1m0aJFikCrDC4ekpEQ1Pxgv3Z09+4uiNSGhjRw7dkw++ugjsn2eeBm1bHmVoQapKZGRDX0+vrMByX/hhRcNdU9r9bJCJbNw4UJp1SrOmrp0uUVQ21DGSoI2OvkeAc14fY+5RyOeO/errF69Ro4cOeK0H6TboKAgmTdvnmzatElmzJgpGzZslOuvv166du2q2kVFRcldd90lJ06ckMmTpwjLen60+/btl1tv7SpIzFTEdev8+fOGnvU8t3L69BnFDOn7qadGy/PPPy+NGzeW7OzNsmzZ+6qOtw6ZmVkyevQY4yUzR3W5YsUKJXVDLy8KlenjA7h0NnTdvIwsFouEhAT7mIKSh+M5h4WFyZkzZ2THjp1FGnz33Xfy4YcfSn5+geTl5RUp0ze+Q0AzXt9h7fFIGRkZ8swzz8hrr71mZYT2nfKja968uRQUFMhnn31mLT506JDBJEKkbdu2Kq99+/bSqFFDxcBZxqtM43Dw4AGJiIiQDh06GHciP/zwgzqbB9QJSJ7m/bBhj0qNGjW8qmIw+0bC3rhxo1JvmHlIv7m5ueatz8+9e/dS9PCiYt61a1/icxpKGhA9+CWXXGIw13yD8e5Q1VGP2Bo+T5484RUDpepcH8qMgGa8ZYbMvxs0a9ZMGaHsqURqJQ+vAM7NmjWVWrVqcVkk/fbbebWERiKmYPfuPYphR0Y24FauuKK5Ujtww2YGGDh6ZFvmTVllTMwX5rV06TK5cOGCmuIf/vAHdXZ6qICC6OgY9Qy/+eYbq+oHtVJKyuPqb4NVy549e9QqpwLI00MaCGjGa4BQmf7DCJBKnc2pTp1CCS0oqJYyhjmrFxZWqDtesmSJHD16VO677z5lUMPYxVI1ylBVDBjQX7ZsyfW6isEZTRWd36NHd0OC3Cnp6enGSuGoIsfESd34ySE6Olq53O3du1dRhLqoU6eOavXCKmLdunUyZkyaktxVBX3wOQKa8foc8vIdEDVBzZo1SxzElGBLqsjSftCg+2T58uUGszkiTz/9jNIbDxo0SDWdO3eukqIeffQReeutNyUtbYy6V4WV6MCLh00bvIhsp1Wrln/peFE14WlRvXp16d27t6F73yQrVnwsf/7zn2XPnkJGbEu/vq4YBDTjrRjcy21UNhuYy2BXg5iqB1d1zDKkpJkz35Dx4ycoj4bu3bvJtde2Vi5oqBjw8cUN7eOPP5ZmhqoDzwizrX+ey0ZVtCFB3nzzTYKxD2MlrdGR4hlgriDIMxPbmVG/JCf3M7McnnFHA7fSJAx66GkddmSTSZ/od1Ez3HNPH2nTJkEZT3/88UfZvXuXTU1RMTBKQ2eRRvrGKwhoxusVGP2nkx9++FEtM0ui6NtvvyupisNyVAx4Q+BOhhcDrmtJSUmydev/Cr6+ixcvkdjYWPWjdtiBkYlnxfr164w2hb6mOTmbJSsr02Bs60tMGzducOmnumVLjrGMHm2M4r3/xIfAYAkj3bo1V9GdnJysjIrBwSHKn9d2NBhvRER9ycsrapi0rcN1z549pH///qVOrVu3ppnLxPNhxYPXCy9FszLud7w4zHvOpaWTujp5FwHNeL2LZ4X3hmSDJOaMENNLwZTYnNXD7chRma2KgXJTwjJ9QteuXavcmFiWU+4o4X9LPVPqRkJfsiRdkpI6lJjatWuv/FRbtYqTvn37yaRJk5T3Bu5R9IN+u02bNgIDcjR2WfN4seAJwjiMaaa0tLFKR1qjRvViqpXmzS+XkydPGvrgHS6Hw32Pl1hpUv/+AwTdrKsOkYgvvzxGvXh3795trQrWkyb9Q9FrzTQuSkunUVX/9zICmvF6GdCK7g6dLL659pJYaGgdsVgsyrEeGrFqnz17VurUqcOtNdWrV1fI37//K2ueeWGvYiC/evUanKwJtQQ3ISEhnJym2bPnyN69+1Q5blndu3dTO+FURikPuHQhZQ8fPkI6deokU6dOE148kZGR0rlz51L24rpa3759lWfAokVvO6yIMTM8PFyV4Tnw9tuLhC3VSJ1soUaNoAp9cGDcunXrqhefrf/u/v37jVVCrpWCiqbTSkgVvtCMN4AfvsViMZa71YvMAB1kbu5WgdEijVKIJHTFFS3k+PHjVqmJTRi7d+8RLN5IddRDl4mOFi+GTz/9lCxrQoJEMsvKyirixXDkyGE5d+6ctR7L1+DgYPnppwJrnqMLmMGMGTMUo6QcvSQMARq4dyfNNQx9I0Y8JgcOHJSbbupcTBItS59ghvtVbOyVsnbtumJNCwryFYOzWCyCIUuMf++9957yZ+bFtXLlSiWRf/DBB0aJb/7zvMExP/93/11HI1c0nY5oqmp5Acl4Bw4cKK+//nox3Vplf3j4ka5atVJZqmFQSFtjx45V+tEpUyZbpz9z5kwVSGbw4PvVzjJiMURHN1PGsPXrM1U9JFNiM5w5c1ZSU1Nl7Ng0eeWViUKf8+bNM5jXAVXPPKBiQIWBVGnmcYbJHz58RGJiohWjI2jMqVOnDV1tFsUu05o1ayQ9fanakkxFlr7E3+Xa3cR2ZrAICgqS2267za1uUlJS5NNP16j4FnXrhsnIkSnC/M3O8N6YPHmyIOnCoJ999hlZvPg95ePMC85isRhqhp1m9XI/P/XUk7JhQ5baFo6kzU7Cd9/9j3z44XLp2LGjw/Ergk6HhFTRTK8xXoJU8/BN48Nnn2UIlmBnuPLHglGF+pxxebnhhhucVbfmw3AwSuDIz9vdWmBzwbgrV34i9Ev/pM2bsxWDIo9Eue2PiQhc1Csp5eZuUUzKZjifXa5YscJYQt+kLNWmrvHaa+MlMTFJWG6bhCBN4gLGPv1q1Syybds2g3Hcp7YOm3U4w4TBYMWKT4TNFEjB6E0xmlFOIvFckKbmzZtfTE+IAYeddKgsYD7UXbZsabF69OMo8VLIyclRGxIsFouwIcM2VoSjNiXlMa8JE152e4MA+tyEhLZiYty+faKSZM1xwSg+vo21nOveve8UVhC8PEz9blxcK7WiMNuV1xlvE2iMi2ttpalt23Zy6623SUZGhsNhK4JOh4RU0UyvMd5//nOy8aNJVD/yU6dOqZ0zSE+OcOXHicEC48qhQ4ekZ89e0qXLLYIE5Ki+bd7AgQOkUaNGajdV/fr1bYus1598stJYat5sLImXqR/0VsPizo8DBgWjevTRR43l8a8ydOgQGTJkiGo3dOhD8ve/P6i2UZ4+fVr44Zo/PM7Qh3RGGYYc1ciPD0i06D8xAsFIYMaOyMWIxuqBepy5t6/Hc0ENwNm+jHt+3N2795CuXW9VUqY946aOq/TKK69adc9Iqt26/U1M9Yerdq7KULmgA3ZVpzzKIiMbCq5coaGhgu8vqpzyGMfTPgOFTk/n6a/tvcZ4mWBCQoJaqiIFseTh4ZJvm1ia8cMikDW6Mf5IqW9bx9k1S20YNgFKsF7Xrl3bWVWVzy4rLvbt28vJmrKyNsihQweFH/n11//Vmo80jRSNccqeUcGQ8FPNy8tTFmtrI33hMQJg/c47/1EBe+gsIiJChg0bJuiVuQ+kRHD1sLAweeSRh1X4SlQf/kh/oNDpj9h5gyavMl4YF3v9+SHBePkB2ROJ/6OZh8S79+K2RjPP1Zktm19++aXaQYUlvF69ek6rszRu2LCRMoA42rFT6+KOI9uALyy/MAwdOXJU7c6ic6JRDRnyoNJfco8hCSbMdYWlSjgwBh+2Iv/2229qdjExMYKOWt0E0GHkyFRhswkrpvnz5/st5YFCp98C6CFhXmW8sbGxkp//oxw9+q2xlD8nl112aRHyYGJt2ybI6tWrDXVBY8UUbd1eilS2u2HZ1qBBA3nrrUWqxGKxGOqGYHXt6BBjGHtCQ+soS/6uXUV37LCMbdKksfJ3RGdrto+JuVxdfv311+rMgShdGIxYupNYvh48eJAinbyMAIa77OxspR5iNYRL2L339vHyKOXbHX8jqF4QPsp3JM96DxQ6PZul/7b2KuONimoiX331lVqKI82GhBSN3HTPPXcbBohCx26YYn4Jbi8mbFFRUYbOtrNy60F3xy4cJN7aLkLyIfHiS8pLINcmjCCqjn79+gnLQX7k+JMyTlxcnDRocJmysOcbLw/8LwcOHKi+yPDNN4epoow1xCqA+aoMffAqAjCDN998U+nZ6ZjnN2DAAI/1vfSlk0bAnxDwGuNFvxsaGio7d+4UluIsyUMNiRNGx4QxzuB+gyO6yRS/sQlbRx1nCannwoULsnjx4iJVnBnvqET4Q/TASN3sRzcTfpV8MgbJmeUWP3bqk4dlHkkLf1W8LjC+hYfXU3OijutUWMrndjIz10tpE25LzK+wtT7ikcASHSMmaLADjhel+XdEnk4agUBHwGuM19Tvok9lWyqeDfxYYLJIrAT5WLVqtfIPhSkiEZdG8naCwgAACvNJREFUv4taoG3btgLDNI1wBQU/KdzDwgpDF6obmwNjNm7cREmvK1eukgULFljT008/LZ063aC2mppMl6aoSYKDg2Xbtu3Wbavjx48XpGukeOowRyRjrp2lJ5540to+qRRbYKGFl5Gz/qpiPnisWrVK0PdaLBa1E8z0PqmKeOg5Vz4EvMZ4YVws0VmGk2BqSKTh4eGCRAczxoBiMsUzDj5L4gheJOUoQ9UwatQoMX1su3S5WVVFQlUXdgdb/W5mZqZi2jBukrP97rwM6MbWAwJXoGPH8uSzi19yGD36KeGzONTzdTLnHkhnTzBCBbRvX+GWYlYh8fHX+sQnFpoDCWNv0srcdfINAl5jvFEX9buQDdP96afjype3ZcuWcuWVVwruQpThOYCkml8K/a7JdFNSRgq+tGZ6441ZyniH0z992ieYO/rBY8eOCZ+OsS+3v0dNQnxaXgZI7GY51mk2GHDfs2dPwYvCZMLkOUqMjX64tIlIXUjSjvqyzTPnHkhnW/rLes3qZs6cOVJQUKDiyI4Zk6ZUWGXtx536gYSxN2l1Byvdxj0EvMJ4YVx169Y1DGu/ewOcPn1KfeHgr3+9TrKyNqg4rpAIE2ZJX5J+N8qQcnEf+/zzz51urECV4WjpDyOzWCyya1ehIY9xXSXq479LLAN7DwjaMc4dd9yhtuGigyTPWSLgdLt2baX0qZ2Y/sbO+qyK+WDeo0cP5fkydeoU6waLqoiFnnPlQ8ArjBc9bM2aQWLrhkW8V3x5CbjC1lUTuhYtYtWHGl3pd1niP/TQUGFnmiPVgGm8s1gsSqo2++aMy1rDhg2VfteWHsqcJdQkvAzsPSCozy672bNnqX3469atJ8tlQp3CLrDSJj5eifuRy06rYCHB1Xkuc+b8W0p62VVBePSUAxwBjxgvwUQ2bfpcBXJu2DBSpk+fJv/4xyQFCUapb7/9VqkYUD0QWCQ7e5OhMviL4ArWp08fh0E8Jk6cKMuXfyBdunQRLNrELCX2Ap0i3b7//jJB14r+mPLXXpsuc+bMFhjkxx9/JP/+9xyJjIwU1BDsHlq6NF3i4lrRvFgimAqBtdlJZ7FYpGXLq6zeCATmzsnZrOaDlH7o0CGDruXF+tAZ3kcAAyi+0++8847wIvP+CP7ZI+qpWbPeEObunxRqqryFQDVPOiIGQIJNMBHiITz+eIrqkn3/t9zS1aom6Nu3X5HgLtR1FMQDIxplpu6KOKvEXqBT/HG7desuxFswy/m0yf33D1bjMJ5tW6579OgpRNCivX0isIxtYG3qm54IZlwHcxw+o8ILxL4Pfe9dBAiQQ0jHxYuXFAvq492R/Ks3VCu9evUSbATh4c53ZPoX1ZoadxHwiPG6O6hupxFwhADG1N69ewmqF6KWOapTWfPYKMKKC/VcZGTDyjpNPa+LCPzOeC9m6JNGoCIQwE5w//33CbE4iHPgKQ3EF7aNC+Jpf+XZHrsEfu4Ef7JYXG+FL086dN++Q0AzXt9hrUdyggBMNy0tTTDETpz4Sqlj+TrpTn1CCBUSkeSc1fGnfKR81Fj4v2P/cLUV3p/o1rS4j4BmvO5jF5AtExPbC9IgfslMAFe6hx9+WPCuwKeYPF8mxh82bLgcP35CMKR6GlwGdQVbvTHu8lFNX87FnbEwIsfFxcnSpYWxo+kDwzFnnSoMgXIfWDPecofYfwZISEiQESMek+joGPU5m4kTJ8q0adOETS2NGzeS5557VohR4SuKMSiNGpUqdepcIp766sbHxwt64dTUkUZ/dQyD6hZfTcOjcXr06K4+E5Seni6EI6WzsLBQTjpVYgQ0463ED9d+angLHD16RLKzs5X/81/+co2MHz9OMWO+GkFgo6ZNm9o3K5d7mO6YMaOFnY1IpmFhdVUkOFyqSpNSU1MV7QsXLpA1a1bLzJkzVBSzoKAg9QHNDRs2lgvd3uyUUKe4RC5ZsqRIt7UuxooukqlvKhUCmvFWqsfpejJsSMG1johtFotF+MaauTmBMvSLBC9y3Yt3StkgwTK7du3aMmjQIHnxxRfKlPr2vVdQjVx99dVqKzfxHEzK2H1I+FDz3h/P0dHR6puEmZlZ1qD7J0+eUDGiWQHY08xqhQh7ycn97IuqzH1lmqhmvJXpaZYwl8cee1zmzZtnqBaaKwMWAVbMJrGxLQTm5WpHoVnX0zM6zQYNGsi+ffuFD0R6MxGWdO3adZ6SWO7t7777LvUcYKQ8B1JycrLw8gsODlH+vLZEwHgjIupLXt4Pttn6OkAR0Iw3QB+cu2S3a9dOIiIiDGPWcWssi6ioKGF77vHjx4XYGO72Xdp2bIQZPPgBIe6xt1OfPvf6/W43vDgIdcoGJHODDue0tLHqhVijRnXrp6ZMTNHDm18vNvP0OXAR0Iw3cJ+dW5TDYAkIZBukCCaArpGIYMePn5D58+epLdhuDaAblYhA3759BayJO+yoMl4N4eHhqggvjbffXqRiErO5Yty4l5QuXBX6w0HT4BYCmvG6BVvgNoqOblZMpUAsCn7UOTlb5LbbblUBhtasWRO4k/RTyjEopqQ8bqwurlSfsbIns6AgX0Vjs1gs6hmJ8Y9YFXPnzpWzZ8/KypUrha33xJU2ivT/AEZAM94AfnjukN64cWP1TTyMbGZ7jFH8sAnyji5x/vz5ZpE+ewmBlJQU4TNP6HHr1g1T7nwYFc3uCSI1efJkQdKFQT/77DOyePF7KireFVe0EIvFotzOzPr6HNgIaMYb2M+vzNS/+uokGTXqCbGVaIkC1r//AOGHP2TI0IAMw4iXAG5oZQbERw3Q5ybYBJRq3z5RkGTN4ZFk4+PbSKtWcSpx3bv3ncr4aKvfjYtrVYovcZi96rO/IqAZr78+mXKii22pjlyt2DFGFDhiHZfT0C67JRQkIT1dVnJQOHDgQCGU4oIF8+WOO253UCPwswiag04+NDRU8P0lXnXgz6pqz0Az3qr9/P1i9iytkeRQd5SVoB07dsiCBQvl+++/L2vTgKmfl3dMwsLChPjSmzfnWL/mEjAT0IQWQ0Az3mKQ6AxfI5CUlChIc+74ECO9r1vn/367IuI2rCNHpgrf/yNqm9a/uw2jXzXUjNevHkfVI4YvlgwfPkJCQkIkMTFRJkwYrzYPoK91lQijiKRcFRAjcllGRkaFfHcuMzOzKkDs8zlqxutzyPWAtgjwxZK9e/dIXl6ecP3kk09JTEx0iR8LjY+P99jIVL9+hC0p+toBAjBegik5KNJZHiCgGa8H4OmmniOA1MpXljEesamAHv/73/8RdnG5Siy7MQhS35Ok22oEKgIBzXgrAnU9phUBW/0u1nqMbH/72x0lBswhyA4uZNaO9IVGIIAQ0Iw3gB5WZSS1WbNmalo7duwUNhfceGNnFTxn48bPxVXavHmzVJTrmyJYHzQCHiCgGa8H4OmmniMA8zxz5oyKP9CgwWVCQPDt27cL22JdpVWrVquAMnxNIz19iTRq1EjF9mW315AhD3pOmO5BI1COCGjGW47g6q5LRoAA7Kmpo4TdcykpI8tsuedzQT179hJ2hSUldRB2e82YMbPkgXUNjUAFIqAZbwWCr4cuRIDddJXdF7dwpvqoEShEQDPeQhz0sUQEdAWNgEbAWwhoxustJHU/GgGNgEaglAhoxltKoHQ1jYBGQCPgLQQ04/UWkhXTjx5VI6ARCEAENOMNwIemSdYIaAQCGwHNeAP7+WnqNQIagQBEQDPecnhoukuNgEZAI+AKgf8HAAD//1NuE6MAAAAGSURBVAMAVQpqSdc0azEAAAAASUVORK5CYII=)

---

  

# 4. Timeline & Milestones (12 Weeks)

|   |   |   |
|---|---|---|
|Phase|Planned Deliverables|Mentor Checkpoint|
|Weeks 1-4 (Data Eng)|clean_data.py script handling outliers/missing values; Export of "Gold Standard" CSV.|☐|
|Weeks 5-8 (Seasonality)|Formal Decomposition Report proving statistical separation of Trend/Season/Residuals.|☐|
|Weeks 9-11 (Forecasting)|Model comparison leaderboard (Prophet vs. SARIMA) & Backtesting results.|☐|
|Week 12 (Executive)|10-slide deck & "One-Click" reproducible pipeline.|☐|

---

  

# 5. Risks & Dependencies

5.1 What's the hardest part technically for you right now? 

Ensuring the residuals are true "White Noise" is the differentiator between students and pros; if patterns exist in the noise, the model is flawed. Additionally, implementing "Sliding Window Validation" correctly to avoid data leakage is complex.

  

5.2 What dependencies or help do you need from mentors? 

- Feedback on model metrics and setup issues.
    
- Guidance on "Great Expectations" data contracts and schema enforcement.
    

---

  

# 6. Evaluation Readiness

6.1 How will you prove that your project "works"? 

- Reproducibility: A requirements.txt and a fixed random seed for all processes.
    
- Project Structure: A professional directory structure (/src, /data, /tests, /notebooks) .
    
- Visual Proof: A Streamlit dashboard where users can interactively test scenarios (e.g., "Heavy Rain" impact).
    

6.2 What success metric or goal will you aim for? 

- Uncertainty Quantification: Forecasts provided as a range (e.g., "500 bikes ±40") rather than a single number.
    
- Scalability: The code must handle a new city's data simply by changing the input file path.
    
- Metric: Low MAPE scores on the "hidden" future window during backtesting.
    

---

  

# 7. Responsibilities

7.1 Responsibilities

|   |   |   |   |
|---|---|---|---|
|Task|Student 1 (Padmanava)|Student 2 (Aparna)|Mentor Notes|
|Data Engineering|☐|☐||
|Decomposition Logic|☐|☐||
|Dashboard UI|☐|☐||
|Documentation|☐|☐||

  

Signatures (Students):  
Mentor Approval:  
Date:

**
