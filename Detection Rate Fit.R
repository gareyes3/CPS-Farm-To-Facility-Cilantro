df = data.frame("Oo" = c(0,5,10,200),
                "rr" = c(0,0.31,0.8,1))



model1<-lm(rr~Oo, data = df)
model1
plot(df$Oo, df$rr)

mult_nls <- nls(rr ~ a*exp(r*Oo), start = list(a = 0.5, r = 0.2), data = df)
coef(mult_nls)

model1
predict(model1,newdata = df2)


alpha <- -20
beta <- -0.05
theta <- 30

# Sample some points along x axis
n <- 100
x <- seq(n)

# Make  y = f(x) + Gaussian_noise 
data.df <- data.frame("Oo" = c(0,5,10,200),
                      "rr" = c(0,0.31,0.8,1))

# plot data
plot(data.df$Oo, data.df$rr)

# Prepare a good inital state
theta.0 <- max(data.df$rr) * 1.1
model.0 <- lm(log(- rr + theta.0) ~ Oo, data=data.df)
alpha.0 <- -exp(coef(model.0)[1])
beta.0 <- coef(model.0)[2]

start <- list(alpha = alpha.0, beta = beta.0, theta = theta.0)

# Fit the model
model <- nls(rr ~ alpha * exp(beta * Oo) + theta , data = data.df, start = start)

# add fitted curve
plot(data.df$Oo, data.df$rr)
lines(data.df$Oo, predict(model, list(x = data.df$rr)), col = 'skyblue', lwd = 3)


df2 = data.frame("Oo" = 0,
                 "rr" = 0)

predict(model,newdata = df2)
