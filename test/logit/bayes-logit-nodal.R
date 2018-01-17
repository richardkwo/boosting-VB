library(BayesLogit)
library(ggplot2)
library(GGally)
library(coda)
library(data.table)
library(gridExtra)

data <- boot::nodal
y = data$r
X = data[,c(-2)]

output <- BayesLogit::logit(y, X, samp=2000, burn=4000,
                            P0 = diag(1, nrow = ncol(X), ncol = ncol(X)))
samples.beta <- as.data.frame(output$beta)
colnames(samples.beta) <- colnames(X)
samples.beta.mcmc <- mcmc(samples.beta)

print(summary(samples.beta.mcmc))
samples.bvb <- read.table("bayes-logit-nodal.out")
colnames(samples.bvb) <- colnames(samples.beta)
samples.beta$method <- "Polya-Gamma"
samples.bvb$method <- "BVI"

reference.method <- "Polya-Gamma"

samples.all <- rbind(samples.beta,
                     samples.bvb)

# samples.all <- rbind(samples.beta,
#                      samples.bvb)

samples.all$method <- as.factor(samples.all$method)
samples.all <- data.table(samples.all)
samples.all.melted <- data.table::melt(samples.all, id.vars = c("method"))

print(ggplot(samples.all.melted, aes(x=variable, y=value, fill=method)) + geom_boxplot())

mean.estimates <- samples.all.melted[, .(mean.est = mean(value)),
                                     by = .(variable, method)]
setkey(mean.estimates, variable)
mean.estimates.true <- mean.estimates[method==reference.method,
                                      .(variable = variable, mean.true = mean.est)]
mean.estimates <- merge(mean.estimates, mean.estimates.true)

fig.mean <- ggplot(mean.estimates[method!=reference.method],
                   aes(x = mean.true, y = mean.est, color = method)) +
    geom_point() + geom_abline(slope = 1, intercept = 0, linetype = 2) +
    xlab("True (MCMC)") + ylab("Estimate") + ggtitle("Mean estimates")
# print(fig.mean)

cov.estimates <- samples.all[, cov(.SD), by = method]
colnames(cov.estimates)[2] <- "cov.est"
cov.estimates[, cov.true := rep(cov.estimates[method==reference.method, cov.est],
                                nlevels(cov.estimates[, method]))]
indicator.diag <- as.vector(diag(ncol(X)))
var.estimates <- cov.estimates[rep(indicator.diag==1, length=.N)]
cov.estimates <- cov.estimates[rep(indicator.diag==0, length=.N)]

fig.var <- ggplot(var.estimates[method!=reference.method],
                  aes(x = cov.true, y = cov.est, color = method)) +
    geom_point() + geom_abline(slope = 1, intercept = 0, linetype = 2) +
    xlab("True (MCMC)") + ylab("Estimate") + ggtitle("Variance estimates")
# print(fig.var)

fig.cov <- ggplot(cov.estimates[method!=reference.method],
                  aes(x = cov.true, y = cov.est, color = method)) +
    geom_point() + geom_abline(slope = 1, intercept = 0, linetype = 2) +
    xlab("True (MCMC)") + ylab("Estimate") + ggtitle("Covariance estimates")
# print(fig.cov)
grid.arrange(fig.mean + theme(legend.position=c(.5, 0.5)),
             fig.var + theme(legend.position="none") + xlim(0.1, 0.39) + ylim(0.1, 0.39),
             fig.cov + theme(legend.position="none"), nrow=1)
