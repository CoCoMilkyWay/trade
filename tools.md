# set python local dev env
alias up="pip install -e /home/chuyin/trade/wtpy"

# mysql
sudo apt update
sudo apt install mysql-server
sudo systemctl start mysql.service
sudo snap install mysql-workbench-community
sudo apt-get install gnome-keyring
sudo snap connect mysql-workbench-community:password-manager-service :password-manager-service
snap connect mysql-workbench-community:ssh-keys

sudo systemctl status mysql
sudo systemctl stop mysql
sudo systemctl start mysql
service --status-all

mysql -u root -p
mysql -u chuyin -p
"C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe"
net stop mysql80
net start mysql80
SHOW DATABASES;
SHOW TABLES;
USE db_name;
CREATE DATABASE IF NOT EXISTS db_name;
DROP DATABASE IF EXISTS db_name;  
CREATE TABLE [IF NOT EXISTS] table_name(Column_name1 datatype, Column_name2 datatype……);
DROP TABLE [IF EXISTS] table_name;
DESCRIBE table_name;

sudo mysql_secure_installation
sudo mysql -u root

USE mysql;
SET GLOBAL validate_password.policy=LOW;
drop user chuyin@localhost;
flush privileges;
CREATE USER 'chuyin'@'localhost' IDENTIFIED BY 'bj721006';
ALTER USER 'chuyin'@'localhost' IDENTIFIED BY 'bj721006';
GRANT ALL PRIVILEGES ON *.* TO 'chuyin'@'localhost';
UPDATE user SET plugin='mysql_native_password' WHERE User='chuyin';
FLUSH PRIVILEGES;
exit;
sudo service mysql restart

SELECT user FROM mysql. user;
SELECT user FROM market. user;

# to show pip/mamba install paths (use pip to install packages not avaliable in mamba)
pip list -v
mamba list -v

mamba update --all
mamba env list
mamba deactivate py_3p6
mamba env remove -n py_3p6
conda remove --name py_3p6 --all

conda config --show
conda config --remove-key proxy_servers
conda clean --all

pip freeze > requirements.txt
cat requirements.txt | xargs -n 1 mamba install
sed 's/==.*//' env/requirements.txt > env/requirements_nameonly.txt

mamba create -n py_3p10 python=3.10 ipykernel
mamba upgrade&update --all

# ===================================================================================
# check blocked ip ports by windows
netsh interface ipv4 show excludedportrange protocol=tcp
