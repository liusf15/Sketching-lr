library(nycflights13)

flt = na.omit(flights)
A = left_join(flt,weather)
B = left_join(A, planes, by="tailnum")
C = left_join(B, airlines, by="carrier")
variate = c("arr_delay","month", "day", "dep_time", "sched_dep_time","dep_delay","arr_time",
              "sched_arr_time","air_time","distance","hour",
            "temp","dewp","humid","wind_dir","wind_speed","wind_gust","precip","pressure",
            "visib","year.y","seats")
dat = C[,variate]
dat = na.omit(dat)

write.csv(dat, "datasets/nycflight.csv")
