echo -e "[INFO]: Installing JAGS"
sudo apt install jags 

echo -e "[INFO]: Installing R packages"
Rscript -e "install.packages('R2jags')"
Rscript -e "install.packages('pacman')"
Rscript -e "library(R2jags)"