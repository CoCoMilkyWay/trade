# env
code . --no-sandbox
git submodule update --init --remote
git config --global http.proxy 127.0.0.1:7890

# install
python get-pip.py
pip freeze > requirements.txt
cat requirements.txt | xargs -n 1 pip install

// virtualenv --no-site-packages prod_env
// source prod_env/bin/activate
// pip install -r requirements.txt
# trade
