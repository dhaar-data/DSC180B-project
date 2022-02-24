install.packages("hextri") # install package if hextri has not been installed yet
library(hextri)

setwd('your path') # set your path here

# Plotting estimators
estimators <- read.csv(file = "inference/estimators.csv", header = TRUE)
colnames(estimators) = c('estimate with true outcome', 
                   'estimate with predicted outcome', 'category', 'feature names')
hextri(`estimate with predicted outcome`~`estimate with true outcome`, 
       data=estimators, class=category, colour=c('blue','lightblue','orange'), 
       style='size', nbins=25, xlim=c(-5, 10), ylim=c(-5, 10))
abline(0,1)
legend('topleft', col=rev(c('blue','lightblue','orange')), pch=15, 
       legend=rev(c('Parametric Bootstrap Postpi', 
                    'Non-Parametric Bootstrap Postpi','No Correction')),
       bty='n')
title(main='Estimators')
dev.copy(png,filename="figures/estimators.png")
dev.off()

# Plotting standard errors
ses <- read.csv(file = "inference/ses.csv", header = TRUE)
colnames(ses) = c('standard error with true outcome', 
                        'standard error with predicted outcome', 'category', 'feature names')
hextri(`standard error with predicted outcome`~`standard error with true outcome`, 
       data=ses, class=category, colour=rev(c('blue','lightblue','orange')), 
       style='size', nbins=25, xlim=c(0.4,1.1), ylim=c(0.4,1.1))
abline(0,1)
legend('topleft', col=rev(c('blue', 'lightblue', 'orange')), pch=15, 
       legend=rev(c('Parametric Bootstrap Postpi', 
                    'Non-Parametric Bootstrap Postpi','No Correction')),
       bty='n')
title(main='Standard Error')
dev.copy(png,filename="figures/ses.png")
dev.off()

# Plotting t-statistic
tstat <- read.csv(file = "inference/t_stat.csv", header = TRUE)
colnames(tstat) = c('statistic with true outcome', 
                  'statistic with predicted outcome', 'category', 'feature names')
hextri(`statistic with predicted outcome`~`statistic with true outcome`, 
       data=tstat, class=category, colour=rev(c('blue','lightblue','orange')), 
       style='size', nbins=25, xlim=c(-10,20), ylim=c(-10,20))
abline(0,1)
legend('topleft', col=rev(c('blue','lightblue','orange')), pch=15, 
       legend=rev(c('Parametric Bootstrap Postpi', 
                    'Non-Parametric Bootstrap Postpi','No correction')),
       bty='n')
title(main='T-Statistics')
dev.copy(png,filename="figures/t_stat.png")
dev.off()

