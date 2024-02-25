# mysql
service --status-all
mysql -u root -p
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

SELECT user FROM mysql. user;
SELECT user FROM market. user;
FLUSH PRIVILEGES;
ALTER USER 'root'@'localhost' IDENTIFIED BY 'bj721006';

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

