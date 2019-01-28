# Title     : a demo example
# Objective : test
# Created by: lemuria
# Created on: 2019-01-29

library(ggplot2)

print(head(diamonds))

g <- ggplot(data=diamonds, aes(x=carat, y=log(price))) + geom_point()
ggsave('diamonds.png', g)

