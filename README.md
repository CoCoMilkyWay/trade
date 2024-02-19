# env
- code . --no-sandbox
- git submodule update --init --remote
- git config --global http.proxy 127.0.0.1:7890

- pip freeze > requirements.txt
- cat requirements.txt | xargs -n 1 mamba install
- sed 's/==.*//' env/requirements.txt > env/requirements_nameonly.txt
- mamba install -v --file env/requirements_nameonly.txt

# primarily use conda/mamba packages
# recommend not to use mamba/conda packages with pip packages
# create old environment for specific packages
# install left-over pip package in conda/mamba environment(not avaliable in system python(pip) env)
mamba update --all
mamba install python=3.6
mamba create -n py_3p6 python=3.6
mamba env list
mamba activate py_3p6
mamba env remove -n py_3p7
conda remove --name py_3p7 --all

mamba install numpy pandas seaborn pandas-datareader 
mamba search PKG --info
# to show pip/mamba install paths (use pip to install packages not avaliable in mamba)
pip list -v
mamba list -v
zipline ingest -b quandl
- export QUANDL_API_KEY="9Q5bVWxqJE-94HKpntUg" ("6y7b4GG74vHE4sssJ8Ef")

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