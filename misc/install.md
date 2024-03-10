# use root to create root WSL (not user WSL)
# ~/.bashrc
export clash_ip="198.18.0.1"
export http_proxy="http://$clash_ip:7890"
export https_proxy="http://$clash_ip:7890"
export ftp_proxy="http://$clash_ip:7890"
export httpProxy="http://$clash_ip:7890"
export httpsProxy="http://$clash_ip:7890"
export ftpProxy="http://$clash_ip:7890"
export HTTP_PROXY="http://$clash_ip:7890"
export HTTPS_PROXY="http://$clash_ip:7890"
alias pip="pip --proxy http://198.18.0.1:7890"
env | grep -i proxy
# use WSL IP(dynamic, use ipconfig to check in windows cmd) as display port to external VCXSRV server
cd /home/work/trade
export DISPLAY=$(ip route list default | awk '{print $3}'):0
export QUANDL_API_KEY="9Q5bVWxqJE-94HKpntUg"
# code .

# APT(ubuntu) proxy
/etc/apt/apt.conf
Acquire::http::proxy "http://198.18.0.1:7890";
Acquire::https::proxy "http://198.18.0.1:7890";
Acquire::ftp::proxy "http://198.18.0.1:7890";

sudo apt install resolvconf
sudo vim /etc/resolvconf/resolv.conf.d/base
nameserver 8.8.8.8
sudo resolvconf -u

# you must update (otherwise proxy not work)
apt upgrade
apt update
# c-libraries / apt system packages
sudo apt install python-dev-is-python3 gfortran pkg-config libfreetype6-dev libatlas-base-dev hdf5-tools python3-pip

# primarily use conda/mamba packages
# recommend not to use mamba/conda packages with pip packages
# create old environment for specific packages
# install left-over pip package in conda/mamba environment(not avaliable in system python(pip) env)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

# env (check email for personal auth token)
git clone --recurse-submodules -j8 https://github.com/CoCoMilkyWay/trade.git
(git clone https://github.com/CoCoMilkyWay/machine-learning-for-trading.git)
(git submodule add https://github.com/CoCoMilkyWay/QuantStudio.git)

git config pull.rebase false
git config --global user.name "CoCoMilkyWay"
git config --global user.email "wangchuyin980321@gmail.com"
git config --global http.proxy http://198.18.0.1:7890
git config --global https.proxy http://198.18.0.1:7890
conda config --set proxy_servers.http http://198.18.0.1:7890
conda config --set proxy_servers.https http://198.18.0.1:7890
conda config --set ssl_verify false

# successful flow installing zipline+alphalens+pyfolio

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
mamba install numpy pandas seaborn pandas-datareader nbconvert mkl-service
mamba search PKG --info
mamba install -v --file misc/req.txt
pip install zipline
mamba uninstall alembic
pip install alembic
pip install iso3166==2.0.2
zipline ingest -b quandl
mamba install alphalens pyfolio PyPortfolioOpt

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

#
