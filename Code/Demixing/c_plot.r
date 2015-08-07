c_1 <- 1:100
c_2 <- 1:100
c_50 <- 13.1
n <- 1.5
s <- 19
s_c <- 19
heeger <- function(c_1, c_2, c_50, n) 
{
  w <- (c_1^n)/(c_50^n + (c_1^2 + c_2^2)^(n/2))
  w
}
df <- expand.grid(c_1=c_1, c_2=c_2)
df$w <- heeger(df$c_1, df$c_2, c_50, n)
p <- qplot(df$c_1, df$c_2, data = df, fill = df$w, geom = "raster")

p + scale_fill_gradient2(name = "w", midpoint = .5)

ma_halpern <- function(x_1, x_2, s, s_c)
{
  norm = dnorm(x_1, sd = s) * dnorm(x_2, sd=s) + dnorm(x_1, sd=s) + dnorm(x_2, sd=s) + dnorm(x_1, mean=x_2, sd=sqrt(2*s^2 + 2*s_c^2))
  w_2 = (x_1/s^2 + x_2/(s^2 + 2*s_c^2))/(1/s^2 + 1/(s^2 + 2*s_c^2))
  num = x_1 * dnorm(x_2, sd=s) + w_2 * dnorm(x_1, mean=x_2, sd=sqrt(2*s^2 + 2*s_c^2))
  w = num/norm
}

estimate_weights <- function(c_1, c_2, s, s_c)
{
  x_1s = rnorm(100, mean=c_1, sd=s)
  x_2s = rnorm(100, mean=c_2, sd=s)
  w = ma_halpern(x_1s, x_2s, s, s_c)
  mean(w/100)
}
vew <- Vectorize(estimate_weights, c("c_1", "c_2"))

df <- expand.grid(c_1=c_1, c_2=c_2)
df$w <- vew(df$c_1, df$c_2, s, s_c)
p <- qplot(df$c_1, df$c_2, data = df, fill = df$w, geom = "raster")
p + scale_fill_gradient2(name = "w", midpoint = .5)

rmse <- function(s)
{
  sig <- s * 100
  df$mh <- vew(df$c_1, df$c_2, sig[1], sig[2])
  sqrt(mean(df$h-df$mh)^2)
}

df <- expand.grid(c_1=c_1, c_2=c_2)
df$h <- heeger(df$c_1, df$c_2, c_50, n)
sigs <- matrix(, 1, 2)
s <- runif(2)
o <- optimx(s, fn=rmse, lower=c(0.001,0.001), upper=c(1,1))
sigs <- c(coef(o[1,]))
for(i in 1:100)
{
  s <- runif(2)
  o <- optimx(s, fn=rmse, lower=c(0.001,0.001), upper=c(1,1))
  sigs <- rbind(sigs, c(coef(o[1,])))
}

c_2w <- seq(0, 100, 10)
dfw <- expand.grid(c_1=c_1, c_2=c_2w)
dfw$h <- heeger(dfw$c_1, dfw$c_2, c_50, n)
dfw$mh <- vew(dfw$c_1, dfw$c_2, 3, 1)
ggplot(dfw, aes(x=c_1, y=h, group=c_2, colour=c_2)) + geom_line()
ggplot(dfw, aes(x=c_1, y=mh, group=c_2, colour=c_2)) + geom_line()

erf <- function(x, mu, s) 2 * pnorm(x * sqrt(2), mean=mu, sd=s) - 1
ma_halpern2 <- function(x_1, x_2, s) 
{
  integrand <- function(c)
  {
    d <- dnorm(x_1, mean=c, sd=s) * erf(c, x_2, s)
    d
  }
  i <- sum(integrand(seq(c-5s, c+5s, s/25)))/100
  w <- .5 + .5 * i
  w
}
ma_halpern2_quad <- function(x_1, x_2, s) 
{
  integrand <- function(c)
  {
    d <- dnorm(x_1, mean=c, sd=s) * erf(c, x_2, s)
    d
  }
  i <- integrate(integrand, lower=x_1-5s, upper=x_1+5s)
  w <- .5 + .5 * i
  w
}

vmh <- Vectorize(ma_halpern2, c("x_1", "x_2"))
estimate_weights2 <- function(c_1, c_2, s)
{
  x_1s = rnorm(100, mean=c_1, sd=s)
  x_2s = rnorm(100, mean=c_2, sd=s)
  w = vmh(x_1s, x_2s, s)
  mean(w)
}
vew2 <- Vectorize(estimate_weights2, c("c_1", "c_2"))

df <- expand.grid(c_1=c_1, c_2=c_2)
df$w <- vew2(df$c_1, df$c_2, s)
p <- qplot(df$c_1, df$c_2, data = df, fill = df$w, geom = "raster")
p + scale_fill_gradient2(name = "w", midpoint = .5)
