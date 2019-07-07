
library("tidyverse")
library("lubridate")
data_dir <- "/Users/krissankaran/.kaggle/competitions/short-term-load-forecasting-challenge"

x <- read_csv(file.path(data_dir, "train.csv")) %>%
  arrange(Month, DayOfTheMonth, Hour, Minute) %>%
  mutate(
    datetime = sprintf("2019-%s-%s %s:%s:00", str_pad(Month, 2, "l", "0"),
                       str_pad(DayOfTheMonth, 2, "l", "0"),
                       str_pad(Hour, 2, "l", "0"),
                       str_pad(Minute, 2, "l", "0"))
  ) %>%
  filter(DayOfTheMonth != 29) %>%
  mutate(
    datetime = as_datetime(datetime),
    numeric_datetime = as.numeric(datetime)
  )

times <- seq(min(x$numeric_datetime), max(x$numeric_datetime), 6000)
interpolated <- list()
for (j in seq_len(ncol(x))) {
  interpolated[[colnames(x)[j]]] <- approx(x$numeric_datetime, x[[j]], times, ties="mean", method = "constant")$y
}

interpolated$datetime <- times
interpolated <- as.data.frame(interpolated)

ggplot(interpolated %>% filter(datetime < 1549446506, datetime > 1547869367)) +
  geom_point(aes(x = datetime, y = Forecast))

## ggplot(x %>% filter(numeric_datetime < 1549446506, numeric_datetime > 1547869367)) +
##   geom_point(aes(x = datetime, y = Forecast))

## ggplot(x) +
##   geom_point(
##     aes(x = Hour, y = Forecast),
##     position = position_jitter(w = 0.1),
##     alpha = 0.2
##   ) +
##   facet_grid(Month ~ DayOfTheWeek)

## plot(x$AirLoad1, x$Forecast)

write_csv(interpolated, "train_interp.csv")
