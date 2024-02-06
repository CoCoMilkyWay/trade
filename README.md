# env
- code . --no-sandbox
- git submodule update --init --remote
- git config --global http.proxy 127.0.0.1:7890

# install
- python get-pip.py
- pip freeze > requirements.txt
- cat requirements.txt | xargs -n 1 pip install

- virtualenv --no-site-packages prod_env
- source prod_env/bin/activate
- mamba install -c conda-forge --file env/requirements_windows.txt  # mamba + pip
- pip install pip_requirements_windows.txt

# trade
- https://www.kaggle.com/competitions/ubiquant-market-prediction/leaderboard
- https://www.kaggle.com/competitions/g-research-crypto-forecasting/leaderboardv
- https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/leaderboard
- Nasdaq API(gmail): 9Q5bVWxqJE-94HKpntUg
- Nasdaq API(qq): 6y7b4GG74vHE4sssJ8Ef

https://cloud.tencent.com/developer/article/1457661