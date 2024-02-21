# env
- code . --no-sandbox
- git submodule update --init --remote
- git config --global http.proxy 127.0.0.1:7890

- pip freeze > requirements.txt
- cat requirements.txt | xargs -n 1 mamba install
- sed 's/==.*//' env/requirements.txt > env/requirements_nameonly.txt
- export QUANDL_API_KEY="9Q5bVWxqJE-94HKpntUg" ("6y7b4GG74vHE4sssJ8Ef")

# primarily use conda/mamba packages
# recommend not to use mamba/conda packages with pip packages
# create old environment for specific packages
# install left-over pip package in conda/mamba environment(not avaliable in system python(pip) env)
mamba update --all
mamba install python=3.6
mamba create -n py_3p6 python=3.6
mamba env list
mamba activate py_3p6
mamba deactivate py_3p6
mamba env remove -n py_3p6
conda remove --name py_3p6 --all

# successful flow installing zipline+alphalens+pyfolio
# c-libraries / apt system packages
sudo apt install libatlas-base-dev python-dev-is-python3 gfortran pkg-config libfreetype6-dev hdf5-tools
# ta-lib
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
sudo ./configure
sudo make
sudo make install

mamba create -n py_3p6 python=3.6 ipykernel
mamba env list
mamba activate py_3p6
mamba install numpy pandas seaborn pandas-datareader nbconvert
mamba search PKG --info
mamba install -v --file req.txt
pip install zipline
mamba uninstall alembic
pip install alembic
pip install iso3166==2.0.2
zipline ingest -b quandl
mamba install alphalens pyfolio

sudo apt-get install sqlitebrowser

from importlib import metadata as importlib_metadata =>
import importlib_metadata

in pyfolio's rolling_fama_french function(line 550) in timeseries.py
~/mambaforge/envs/py_3p6/lib/python3.6/site-packages/pyfolio/timeseries.py
change to this so that A, B has same dimensions:
    A = factor_returns[beg:end]
    B = returns[beg:end]
    idx = A.index.intersection(B.index)
    A = A.loc[idx]
    B = B.loc[idx]

# to show pip/mamba install paths (use pip to install packages not avaliable in mamba)
pip list -v
mamba list -v

mamba create -n py_3p10 python=3.10 ipykernel
mamba install numpy pandas seaborn pandas-datareader nbconvert zipline-reloaded alphalens-reloaded pyfolio-reloaded
mamba upgrade&update --all

# trade
- https://bigquant.com/trading/list
- https://www.kaggle.com/competitions/ubiquant-market-prediction/leaderboard
- https://www.kaggle.com/competitions/g-research-crypto-forecasting/leaderboardv
- https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/leaderboard


# textbook
    beginner:
        systematic trading -robert carver ***
        trading systems and methods -perry kaufman *
        advances in financial machine learning -marcos lopez ***
    risk managements:
        the leverage space trading model **
        mathematics of money management -ralph vince
    indicators:
        rocket science for trades -john ehlers
        cybernetic analysis for stocks and futures -john ehlers
        cycle analytics for traders -john ehlers ***
        statistically sound indicators for financial market prediction -timothy masters **
    stratergies:
        the universal tatics of successful trend trading -brent penfold
        stock on the move -andrew clenow
        cybernetic trading strategies -murray ruggireo
    system dev:
        testing and tuning market trading algorithms -timothy masters **
        permutation and randomization tests for trading system developments -timothy masters **
    misc:
        numerical recipes -william press
        assessing and improving prediction and classification -timothy masters
        data-driven science and engineering -steve brunton
    niche:
        technical analysis for algorithm pattern recognition -prodromos
        detecting regime change in computational finance -jun chen
        trading on sentiment -richard peterson

# misc
https://cloud.tencent.com/developer/article/1457661
https://github.com/jiangtiantu/factorhub