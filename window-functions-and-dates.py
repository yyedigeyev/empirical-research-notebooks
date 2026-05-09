# Databricks notebook source
# MAGIC %md
# MAGIC # Week 4: Window Functions & Date/Time Analysis
# MAGIC
# MAGIC This week combines window functions with date/time operations — they pair naturally since window functions excel at time-based analysis (comparing periods, running totals, trends).
# MAGIC
# MAGIC **Datasets:** flights (date/time columns), diamonds (for non-time window examples)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Date and Time Basics
# MAGIC
# MAGIC The flights table has date/time data spread across columns: `year`, `month`, `day`, `hour`, `minute`. Let's explore extracting and manipulating date/time components.

# COMMAND ----------

# First, let's see what date/time columns we have
flights = spark.table("flights")
flights.select("year", "month", "day", "hour", "minute", "dep_time").limit(10).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### SQL: Extracting Date Components

# COMMAND ----------

# MAGIC %sql
# MAGIC -- The flights table already has year, month, day as separate columns
# MAGIC -- Let's create a proper date column and extract components from it
# MAGIC SELECT
# MAGIC     year, month, day,
# MAGIC     MAKE_DATE(year, month, day) AS flight_date,
# MAGIC     DAYOFWEEK(MAKE_DATE(year, month, day)) AS day_of_week,
# MAGIC     DAYOFYEAR(MAKE_DATE(year, month, day)) AS day_of_year,
# MAGIC     WEEKOFYEAR(MAKE_DATE(year, month, day)) AS week_of_year,
# MAGIC     QUARTER(MAKE_DATE(year, month, day)) AS quarter
# MAGIC FROM flights
# MAGIC WHERE year IS NOT NULL AND month IS NOT NULL AND day IS NOT NULL
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC -- DATE_TRUNC: Truncate to a specific time period (great for grouping)
# MAGIC SELECT
# MAGIC     MAKE_DATE(year, month, day) AS flight_date,
# MAGIC     DATE_TRUNC('MONTH', MAKE_DATE(year, month, day)) AS month_start,
# MAGIC     DATE_TRUNC('QUARTER', MAKE_DATE(year, month, day)) AS quarter_start,
# MAGIC     DATE_TRUNC('WEEK', MAKE_DATE(year, month, day)) AS week_start
# MAGIC FROM flights
# MAGIC WHERE year IS NOT NULL AND month IS NOT NULL AND day IS NOT NULL
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Date arithmetic: DATEDIFF, DATE_ADD, DATE_SUB
# MAGIC SELECT
# MAGIC     MAKE_DATE(year, month, day) AS flight_date,
# MAGIC     DATE_ADD(MAKE_DATE(year, month, day), 7) AS one_week_later,
# MAGIC     DATE_SUB(MAKE_DATE(year, month, day), 30) AS thirty_days_before,
# MAGIC     DATEDIFF(MAKE_DATE(2013, 12, 31), MAKE_DATE(year, month, day)) AS days_until_year_end
# MAGIC FROM flights
# MAGIC WHERE year IS NOT NULL AND month IS NOT NULL AND day IS NOT NULL
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC -- DATE_FORMAT: Format dates for display
# MAGIC SELECT
# MAGIC     MAKE_DATE(year, month, day) AS flight_date,
# MAGIC     DATE_FORMAT(MAKE_DATE(year, month, day), 'EEEE') AS day_name,
# MAGIC     DATE_FORMAT(MAKE_DATE(year, month, day), 'MMMM') AS month_name,
# MAGIC     DATE_FORMAT(MAKE_DATE(year, month, day), 'MMM dd, yyyy') AS formatted_date
# MAGIC FROM flights
# MAGIC WHERE year IS NOT NULL AND month IS NOT NULL AND day IS NOT NULL
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ### Python: Date Functions

# COMMAND ----------

from pyspark.sql.functions import (
    col, make_date, year, month, dayofmonth, dayofweek, dayofyear,
    weekofyear, quarter, date_trunc, date_add, date_sub, datediff,
    date_format, to_date, current_date
)

# Create a proper date column
flights_with_date = flights.withColumn(
    "flight_date",
    make_date(col("year"), col("month"), col("day"))
)

# Extract various date components
flights_components = flights_with_date.select(
    col("flight_date"),
    dayofweek("flight_date").alias("day_of_week"),
    dayofyear("flight_date").alias("day_of_year"),
    weekofyear("flight_date").alias("week_of_year"),
    quarter("flight_date").alias("quarter")
)
display(flights_components.limit(10))

# COMMAND ----------

# DATE_TRUNC for grouping by period
flights_truncated = flights_with_date.select(
    col("flight_date"),
    date_trunc("month", col("flight_date")).alias("month_start"),
    date_trunc("quarter", col("flight_date")).alias("quarter_start"),
    date_trunc("week", col("flight_date")).alias("week_start")
)
display(flights_truncated.limit(10))

# COMMAND ----------

from pyspark.sql.functions import lit

# Date arithmetic
flights_arithmetic = flights_with_date.select(
    col("flight_date"),
    date_add(col("flight_date"), 7).alias("one_week_later"),
    date_sub(col("flight_date"), 30).alias("thirty_days_before"),
    datediff(lit("2013-12-31"), col("flight_date")).alias("days_until_year_end")
)
display(flights_arithmetic.limit(10))

# COMMAND ----------

# Date formatting
flights_formatted = flights_with_date.select(
    col("flight_date"),
    date_format(col("flight_date"), "EEEE").alias("day_name"),
    date_format(col("flight_date"), "MMMM").alias("month_name"),
    date_format(col("flight_date"), "MMM dd, yyyy").alias("formatted_date")
)
display(flights_formatted.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Time-Based Aggregations
# MAGIC
# MAGIC Now let's use date/time functions for meaningful analysis of flight patterns.

# COMMAND ----------

# MAGIC %md
# MAGIC ### SQL: Time-Based Analysis

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Flights per month
# MAGIC SELECT
# MAGIC     month,
# MAGIC     COUNT(*) AS num_flights,
# MAGIC     ROUND(AVG(arr_delay), 2) AS avg_arrival_delay
# MAGIC FROM flights
# MAGIC WHERE arr_delay IS NOT NULL
# MAGIC GROUP BY month
# MAGIC ORDER BY month

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Average delays by hour of day
# MAGIC -- Note: hour column contains scheduled departure hour
# MAGIC SELECT
# MAGIC     hour AS departure_hour,
# MAGIC     COUNT(*) AS num_flights,
# MAGIC     ROUND(AVG(dep_delay), 2) AS avg_dep_delay,
# MAGIC     ROUND(AVG(arr_delay), 2) AS avg_arr_delay
# MAGIC FROM flights
# MAGIC WHERE hour IS NOT NULL AND dep_delay IS NOT NULL
# MAGIC GROUP BY hour
# MAGIC ORDER BY hour

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Weekday vs Weekend comparison
# MAGIC -- DAYOFWEEK: 1=Sunday, 2=Monday, ... 7=Saturday
# MAGIC SELECT
# MAGIC     CASE
# MAGIC         WHEN DAYOFWEEK(MAKE_DATE(year, month, day)) IN (1, 7) THEN 'Weekend'
# MAGIC         ELSE 'Weekday'
# MAGIC     END AS day_type,
# MAGIC     COUNT(*) AS num_flights,
# MAGIC     ROUND(AVG(dep_delay), 2) AS avg_dep_delay,
# MAGIC     ROUND(AVG(arr_delay), 2) AS avg_arr_delay
# MAGIC FROM flights
# MAGIC WHERE year IS NOT NULL AND arr_delay IS NOT NULL
# MAGIC GROUP BY day_type

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Busiest travel days
# MAGIC SELECT
# MAGIC     MAKE_DATE(year, month, day) AS flight_date,
# MAGIC     DATE_FORMAT(MAKE_DATE(year, month, day), 'EEEE') AS day_name,
# MAGIC     COUNT(*) AS num_flights
# MAGIC FROM flights
# MAGIC WHERE year IS NOT NULL
# MAGIC GROUP BY year, month, day
# MAGIC ORDER BY num_flights DESC
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ### Python: Time-Based Analysis

# COMMAND ----------

from pyspark.sql.functions import count, avg, round, when

# Flights per month
monthly_stats = (
    flights
    .filter(col("arr_delay").isNotNull())
    .groupBy("month")
    .agg(
        count("*").alias("num_flights"),
        round(avg("arr_delay"), 2).alias("avg_arrival_delay")
    )
    .orderBy("month")
)
display(monthly_stats)

# COMMAND ----------

# Average delays by hour of day
hourly_delays = (
    flights
    .filter(col("hour").isNotNull() & col("dep_delay").isNotNull())
    .groupBy("hour")
    .agg(
        count("*").alias("num_flights"),
        round(avg("dep_delay"), 2).alias("avg_dep_delay"),
        round(avg("arr_delay"), 2).alias("avg_arr_delay")
    )
    .orderBy("hour")
)
display(hourly_delays)

# COMMAND ----------

# Weekday vs Weekend comparison
flights_with_daytype = flights_with_date.withColumn(
    "day_type",
    when(dayofweek("flight_date").isin(1, 7), "Weekend").otherwise("Weekday")
)

daytype_comparison = (
    flights_with_daytype
    .filter(col("arr_delay").isNotNull())
    .groupBy("day_type")
    .agg(
        count("*").alias("num_flights"),
        round(avg("dep_delay"), 2).alias("avg_dep_delay"),
        round(avg("arr_delay"), 2).alias("avg_arr_delay")
    )
)
display(daytype_comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Window Functions Introduction
# MAGIC
# MAGIC **Key Concept:** Window functions perform calculations across rows related to the current row — without collapsing rows like GROUP BY does.
# MAGIC
# MAGIC - `GROUP BY` reduces rows: many rows → one row per group
# MAGIC - `OVER()` keeps all rows: adds a new column with the calculation result
# MAGIC
# MAGIC **Syntax:**
# MAGIC ```sql
# MAGIC function() OVER (
# MAGIC     PARTITION BY column    -- optional: groups within the window
# MAGIC     ORDER BY column        -- optional: ordering within each partition
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Compare GROUP BY vs Window Function
# MAGIC
# MAGIC -- GROUP BY: One row per carrier (collapses data)
# MAGIC SELECT carrier, ROUND(AVG(arr_delay), 2) AS avg_delay
# MAGIC FROM flights
# MAGIC WHERE arr_delay IS NOT NULL
# MAGIC GROUP BY carrier
# MAGIC LIMIT 5

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Window Function: Every row kept, with carrier average added
# MAGIC SELECT
# MAGIC     carrier,
# MAGIC     flight,
# MAGIC     arr_delay,
# MAGIC     ROUND(AVG(arr_delay) OVER (PARTITION BY carrier), 2) AS carrier_avg_delay
# MAGIC FROM flights
# MAGIC WHERE arr_delay IS NOT NULL
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ### Python: Window Functions Basics

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import avg, round

# Define a window partitioned by carrier
carrier_window = Window.partitionBy("carrier")

# Add carrier average as a new column (keeps all rows)
flights_with_avg = (
    flights
    .filter(col("arr_delay").isNotNull())
    .withColumn(
        "carrier_avg_delay",
        round(avg("arr_delay").over(carrier_window), 2)
    )
    .select("carrier", "flight", "arr_delay", "carrier_avg_delay")
)
display(flights_with_avg.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Ranking Functions
# MAGIC
# MAGIC | Function | Behavior |
# MAGIC |----------|----------|
# MAGIC | ROW_NUMBER() | Unique sequential numbers (1, 2, 3, 4...) |
# MAGIC | RANK() | Same rank for ties, gaps after (1, 1, 3, 4...) |
# MAGIC | DENSE_RANK() | Same rank for ties, no gaps (1, 1, 2, 3...) |
# MAGIC | NTILE(n) | Divide into n buckets (1, 1, 2, 2, 3, 3...) |

# COMMAND ----------

# MAGIC %md
# MAGIC ### SQL: Ranking Functions

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Rank flights by delay within each carrier
# MAGIC SELECT
# MAGIC     carrier,
# MAGIC     flight,
# MAGIC     arr_delay,
# MAGIC     ROW_NUMBER() OVER (PARTITION BY carrier ORDER BY arr_delay DESC) AS row_num,
# MAGIC     RANK() OVER (PARTITION BY carrier ORDER BY arr_delay DESC) AS rank,
# MAGIC     DENSE_RANK() OVER (PARTITION BY carrier ORDER BY arr_delay DESC) AS dense_rank
# MAGIC FROM flights
# MAGIC WHERE arr_delay IS NOT NULL
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Find the top 3 worst delays for each carrier
# MAGIC WITH ranked_flights AS (
# MAGIC     SELECT
# MAGIC         carrier,
# MAGIC         flight,
# MAGIC         month,
# MAGIC         day,
# MAGIC         arr_delay,
# MAGIC         ROW_NUMBER() OVER (PARTITION BY carrier ORDER BY arr_delay DESC) AS delay_rank
# MAGIC     FROM flights
# MAGIC     WHERE arr_delay IS NOT NULL
# MAGIC )
# MAGIC SELECT *
# MAGIC FROM ranked_flights
# MAGIC WHERE delay_rank <= 3
# MAGIC ORDER BY carrier, delay_rank

# COMMAND ----------

# MAGIC %sql
# MAGIC -- NTILE: Divide diamonds into price quartiles within each cut
# MAGIC SELECT
# MAGIC     cut,
# MAGIC     carat,
# MAGIC     price,
# MAGIC     NTILE(4) OVER (PARTITION BY cut ORDER BY price) AS price_quartile
# MAGIC FROM diamonds
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Rank diamonds by price within each cut category
# MAGIC SELECT
# MAGIC     cut,
# MAGIC     carat,
# MAGIC     price,
# MAGIC     DENSE_RANK() OVER (PARTITION BY cut ORDER BY price DESC) AS price_rank
# MAGIC FROM diamonds
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ### Python: Ranking Functions

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, rank, dense_rank, ntile

# Window partitioned by carrier, ordered by delay descending
carrier_delay_window = Window.partitionBy("carrier").orderBy(col("arr_delay").desc())

# Add ranking columns
flights_ranked = (
    flights
    .filter(col("arr_delay").isNotNull())
    .withColumn("row_num", row_number().over(carrier_delay_window))
    .withColumn("rank", rank().over(carrier_delay_window))
    .withColumn("dense_rank", dense_rank().over(carrier_delay_window))
    .select("carrier", "flight", "arr_delay", "row_num", "rank", "dense_rank")
)
display(flights_ranked.limit(20))

# COMMAND ----------

# Top 3 worst delays per carrier
top_delays = (
    flights
    .filter(col("arr_delay").isNotNull())
    .withColumn("delay_rank", row_number().over(carrier_delay_window))
    .filter(col("delay_rank") <= 3)
    .select("carrier", "flight", "month", "day", "arr_delay", "delay_rank")
    .orderBy("carrier", "delay_rank")
)
display(top_delays)

# COMMAND ----------

# NTILE: Divide diamonds into quartiles by price within each cut
diamonds = spark.table("diamonds")
cut_price_window = Window.partitionBy("cut").orderBy("price")

diamonds_quartiles = (
    diamonds
    .withColumn("price_quartile", ntile(4).over(cut_price_window))
    .select("cut", "carat", "price", "price_quartile")
)
display(diamonds_quartiles.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5: Analytic Functions
# MAGIC
# MAGIC Analytic functions let you access values from other rows:
# MAGIC
# MAGIC | Function | Description |
# MAGIC |----------|-------------|
# MAGIC | LAG(col, n) | Value from n rows before |
# MAGIC | LEAD(col, n) | Value from n rows after |
# MAGIC | FIRST_VALUE(col) | First value in the window |
# MAGIC | LAST_VALUE(col) | Last value in the window |

# COMMAND ----------

# MAGIC %md
# MAGIC ### SQL: LAG and LEAD

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Compare each flight's delay to the previous flight for the same carrier
# MAGIC SELECT
# MAGIC     carrier,
# MAGIC     month,
# MAGIC     day,
# MAGIC     flight,
# MAGIC     arr_delay,
# MAGIC     LAG(arr_delay, 1) OVER (PARTITION BY carrier ORDER BY month, day, dep_time) AS prev_flight_delay,
# MAGIC     arr_delay - LAG(arr_delay, 1) OVER (PARTITION BY carrier ORDER BY month, day, dep_time) AS delay_change
# MAGIC FROM flights
# MAGIC WHERE arr_delay IS NOT NULL
# MAGIC LIMIT 15

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Month-over-month comparison: average delay by carrier and month
# MAGIC WITH monthly_delays AS (
# MAGIC     SELECT
# MAGIC         carrier,
# MAGIC         month,
# MAGIC         ROUND(AVG(arr_delay), 2) AS avg_delay
# MAGIC     FROM flights
# MAGIC     WHERE arr_delay IS NOT NULL
# MAGIC     GROUP BY carrier, month
# MAGIC )
# MAGIC SELECT
# MAGIC     carrier,
# MAGIC     month,
# MAGIC     avg_delay,
# MAGIC     LAG(avg_delay, 1) OVER (PARTITION BY carrier ORDER BY month) AS prev_month_delay,
# MAGIC     ROUND(avg_delay - LAG(avg_delay, 1) OVER (PARTITION BY carrier ORDER BY month), 2) AS mom_change
# MAGIC FROM monthly_delays
# MAGIC ORDER BY carrier, month

# COMMAND ----------

# MAGIC %sql
# MAGIC -- FIRST_VALUE and LAST_VALUE: Compare to best and worst in group
# MAGIC SELECT
# MAGIC     carrier,
# MAGIC     flight,
# MAGIC     arr_delay,
# MAGIC     FIRST_VALUE(arr_delay) OVER (
# MAGIC         PARTITION BY carrier
# MAGIC         ORDER BY arr_delay
# MAGIC         ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
# MAGIC     ) AS best_delay,
# MAGIC     LAST_VALUE(arr_delay) OVER (
# MAGIC         PARTITION BY carrier
# MAGIC         ORDER BY arr_delay
# MAGIC         ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
# MAGIC     ) AS worst_delay
# MAGIC FROM flights
# MAGIC WHERE arr_delay IS NOT NULL
# MAGIC LIMIT 15

# COMMAND ----------

# MAGIC %md
# MAGIC ### Python: LAG and LEAD

# COMMAND ----------

from pyspark.sql.functions import lag, lead, first, last

# Window ordered by date and time within each carrier
time_window = Window.partitionBy("carrier").orderBy("month", "day", "dep_time")

# Compare to previous flight
flights_with_prev = (
    flights
    .filter(col("arr_delay").isNotNull())
    .withColumn("prev_flight_delay", lag("arr_delay", 1).over(time_window))
    .withColumn("delay_change", col("arr_delay") - col("prev_flight_delay"))
    .select("carrier", "month", "day", "flight", "arr_delay", "prev_flight_delay", "delay_change")
)
display(flights_with_prev.limit(15))

# COMMAND ----------

# Month-over-month delay comparison
monthly_delays = (
    flights
    .filter(col("arr_delay").isNotNull())
    .groupBy("carrier", "month")
    .agg(round(avg("arr_delay"), 2).alias("avg_delay"))
)

month_window = Window.partitionBy("carrier").orderBy("month")

mom_comparison = (
    monthly_delays
    .withColumn("prev_month_delay", lag("avg_delay", 1).over(month_window))
    .withColumn("mom_change", round(col("avg_delay") - col("prev_month_delay"), 2))
    .orderBy("carrier", "month")
)
display(mom_comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 6: Running Calculations
# MAGIC
# MAGIC Window functions with aggregate functions create running totals, moving averages, and cumulative counts.

# COMMAND ----------

# MAGIC %md
# MAGIC ### SQL: Running Totals and Moving Averages

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Running total: cumulative flights per carrier throughout the year
# MAGIC WITH daily_flights AS (
# MAGIC     SELECT
# MAGIC         carrier,
# MAGIC         MAKE_DATE(year, month, day) AS flight_date,
# MAGIC         COUNT(*) AS num_flights
# MAGIC     FROM flights
# MAGIC     WHERE year IS NOT NULL
# MAGIC     GROUP BY carrier, year, month, day
# MAGIC )
# MAGIC SELECT
# MAGIC     carrier,
# MAGIC     flight_date,
# MAGIC     num_flights,
# MAGIC     SUM(num_flights) OVER (
# MAGIC         PARTITION BY carrier
# MAGIC         ORDER BY flight_date
# MAGIC         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
# MAGIC     ) AS cumulative_flights
# MAGIC FROM daily_flights
# MAGIC ORDER BY carrier, flight_date
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 7-day moving average of delays
# MAGIC WITH daily_delays AS (
# MAGIC     SELECT
# MAGIC         MAKE_DATE(year, month, day) AS flight_date,
# MAGIC         ROUND(AVG(arr_delay), 2) AS avg_delay
# MAGIC     FROM flights
# MAGIC     WHERE arr_delay IS NOT NULL AND year IS NOT NULL
# MAGIC     GROUP BY year, month, day
# MAGIC )
# MAGIC SELECT
# MAGIC     flight_date,
# MAGIC     avg_delay,
# MAGIC     ROUND(AVG(avg_delay) OVER (
# MAGIC         ORDER BY flight_date
# MAGIC         ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
# MAGIC     ), 2) AS moving_avg_7day
# MAGIC FROM daily_delays
# MAGIC ORDER BY flight_date
# MAGIC LIMIT 30

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Running percentage of total
# MAGIC WITH monthly_counts AS (
# MAGIC     SELECT
# MAGIC         month,
# MAGIC         COUNT(*) AS num_flights
# MAGIC     FROM flights
# MAGIC     GROUP BY month
# MAGIC )
# MAGIC SELECT
# MAGIC     month,
# MAGIC     num_flights,
# MAGIC     SUM(num_flights) OVER (ORDER BY month) AS cumulative_flights,
# MAGIC     ROUND(
# MAGIC         100.0 * SUM(num_flights) OVER (ORDER BY month) /
# MAGIC         SUM(num_flights) OVER (), 2
# MAGIC     ) AS cumulative_pct
# MAGIC FROM monthly_counts
# MAGIC ORDER BY month

# COMMAND ----------

# MAGIC %md
# MAGIC ### Python: Running Totals and Moving Averages

# COMMAND ----------

from pyspark.sql.functions import sum as spark_sum

# Cumulative flights per carrier
daily_flights = (
    flights_with_date
    .groupBy("carrier", "flight_date")
    .agg(count("*").alias("num_flights"))
)

cumulative_window = Window.partitionBy("carrier").orderBy("flight_date").rowsBetween(Window.unboundedPreceding, Window.currentRow)

cumulative_flights = (
    daily_flights
    .withColumn("cumulative_flights", spark_sum("num_flights").over(cumulative_window))
    .orderBy("carrier", "flight_date")
)
display(cumulative_flights.limit(20))

# COMMAND ----------

# 7-day moving average
daily_delays = (
    flights_with_date
    .filter(col("arr_delay").isNotNull())
    .groupBy("flight_date")
    .agg(round(avg("arr_delay"), 2).alias("avg_delay"))
)

# Window for 7-day moving average (6 preceding rows + current row)
moving_avg_window = Window.orderBy("flight_date").rowsBetween(-6, 0)

moving_avg_delays = (
    daily_delays
    .withColumn("moving_avg_7day", round(avg("avg_delay").over(moving_avg_window), 2))
    .orderBy("flight_date")
)
display(moving_avg_delays.limit(30))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 7: Combining Date/Time with Windows
# MAGIC
# MAGIC The real power comes from combining date/time functions with window functions for time-series analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ### SQL: Time-Series Window Analysis

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Compare each month to the previous month
# MAGIC WITH monthly_stats AS (
# MAGIC     SELECT
# MAGIC         month,
# MAGIC         COUNT(*) AS num_flights,
# MAGIC         ROUND(AVG(arr_delay), 2) AS avg_delay
# MAGIC     FROM flights
# MAGIC     WHERE arr_delay IS NOT NULL
# MAGIC     GROUP BY month
# MAGIC )
# MAGIC SELECT
# MAGIC     month,
# MAGIC     num_flights,
# MAGIC     avg_delay,
# MAGIC     LAG(num_flights, 1) OVER (ORDER BY month) AS prev_month_flights,
# MAGIC     LAG(avg_delay, 1) OVER (ORDER BY month) AS prev_month_delay,
# MAGIC     ROUND(100.0 * (num_flights - LAG(num_flights, 1) OVER (ORDER BY month)) /
# MAGIC           NULLIF(LAG(num_flights, 1) OVER (ORDER BY month), 0), 2) AS pct_change_flights
# MAGIC FROM monthly_stats
# MAGIC ORDER BY month

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Find the worst delay day for each airline
# MAGIC WITH daily_carrier_delays AS (
# MAGIC     SELECT
# MAGIC         carrier,
# MAGIC         MAKE_DATE(year, month, day) AS flight_date,
# MAGIC         ROUND(AVG(arr_delay), 2) AS avg_delay,
# MAGIC         COUNT(*) AS num_flights
# MAGIC     FROM flights
# MAGIC     WHERE arr_delay IS NOT NULL AND year IS NOT NULL
# MAGIC     GROUP BY carrier, year, month, day
# MAGIC ),
# MAGIC ranked_days AS (
# MAGIC     SELECT
# MAGIC         *,
# MAGIC         ROW_NUMBER() OVER (PARTITION BY carrier ORDER BY avg_delay DESC) AS delay_rank
# MAGIC     FROM daily_carrier_delays
# MAGIC )
# MAGIC SELECT
# MAGIC     carrier,
# MAGIC     flight_date,
# MAGIC     avg_delay,
# MAGIC     num_flights
# MAGIC FROM ranked_days
# MAGIC WHERE delay_rank = 1
# MAGIC ORDER BY avg_delay DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Rank months by total delays for each carrier
# MAGIC WITH monthly_delays AS (
# MAGIC     SELECT
# MAGIC         carrier,
# MAGIC         month,
# MAGIC         SUM(arr_delay) AS total_delay,
# MAGIC         COUNT(*) AS num_flights,
# MAGIC         ROUND(AVG(arr_delay), 2) AS avg_delay
# MAGIC     FROM flights
# MAGIC     WHERE arr_delay IS NOT NULL
# MAGIC     GROUP BY carrier, month
# MAGIC )
# MAGIC SELECT
# MAGIC     carrier,
# MAGIC     month,
# MAGIC     total_delay,
# MAGIC     avg_delay,
# MAGIC     DENSE_RANK() OVER (PARTITION BY carrier ORDER BY total_delay DESC) AS delay_rank
# MAGIC FROM monthly_delays
# MAGIC ORDER BY carrier, delay_rank

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Identify flights that were worse than the carrier's monthly average
# MAGIC SELECT
# MAGIC     carrier,
# MAGIC     month,
# MAGIC     flight,
# MAGIC     arr_delay,
# MAGIC     ROUND(AVG(arr_delay) OVER (PARTITION BY carrier, month), 2) AS carrier_month_avg,
# MAGIC     ROUND(arr_delay - AVG(arr_delay) OVER (PARTITION BY carrier, month), 2) AS vs_avg
# MAGIC FROM flights
# MAGIC WHERE arr_delay IS NOT NULL
# MAGIC ORDER BY vs_avg DESC
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ### Python: Time-Series Window Analysis

# COMMAND ----------

# Compare each month to previous month with percentage change
monthly_stats = (
    flights
    .filter(col("arr_delay").isNotNull())
    .groupBy("month")
    .agg(
        count("*").alias("num_flights"),
        round(avg("arr_delay"), 2).alias("avg_delay")
    )
)

month_window = Window.orderBy("month")

monthly_comparison = (
    monthly_stats
    .withColumn("prev_month_flights", lag("num_flights", 1).over(month_window))
    .withColumn("prev_month_delay", lag("avg_delay", 1).over(month_window))
    .withColumn(
        "pct_change_flights",
        round(100.0 * (col("num_flights") - col("prev_month_flights")) / col("prev_month_flights"), 2)
    )
    .orderBy("month")
)
display(monthly_comparison)

# COMMAND ----------

# Worst delay day for each carrier
from pyspark.sql.functions import sum as spark_sum

daily_carrier_delays = (
    flights_with_date
    .filter(col("arr_delay").isNotNull())
    .groupBy("carrier", "flight_date")
    .agg(
        round(avg("arr_delay"), 2).alias("avg_delay"),
        count("*").alias("num_flights")
    )
)

carrier_date_window = Window.partitionBy("carrier").orderBy(col("avg_delay").desc())

worst_days = (
    daily_carrier_delays
    .withColumn("delay_rank", row_number().over(carrier_date_window))
    .filter(col("delay_rank") == 1)
    .select("carrier", "flight_date", "avg_delay", "num_flights")
    .orderBy(col("avg_delay").desc())
)
display(worst_days)

# COMMAND ----------

# Flights worse than carrier's monthly average
carrier_month_window = Window.partitionBy("carrier", "month")

flights_vs_avg = (
    flights
    .filter(col("arr_delay").isNotNull())
    .withColumn("carrier_month_avg", round(avg("arr_delay").over(carrier_month_window), 2))
    .withColumn("vs_avg", round(col("arr_delay") - col("carrier_month_avg"), 2))
    .select("carrier", "month", "flight", "arr_delay", "carrier_month_avg", "vs_avg")
    .orderBy(col("vs_avg").desc())
)
display(flights_vs_avg.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick Reference
# MAGIC
# MAGIC ### Date/Time Functions
# MAGIC
# MAGIC | Operation | SQL | Python |
# MAGIC |-----------|-----|--------|
# MAGIC | Create date | `MAKE_DATE(y, m, d)` | `make_date(y, m, d)` |
# MAGIC | Extract year | `YEAR(date)` | `year(col)` |
# MAGIC | Extract month | `MONTH(date)` | `month(col)` |
# MAGIC | Extract day | `DAY(date)` | `dayofmonth(col)` |
# MAGIC | Day of week | `DAYOFWEEK(date)` | `dayofweek(col)` |
# MAGIC | Truncate to period | `DATE_TRUNC('MONTH', date)` | `date_trunc('month', col)` |
# MAGIC | Add days | `DATE_ADD(date, n)` | `date_add(col, n)` |
# MAGIC | Subtract days | `DATE_SUB(date, n)` | `date_sub(col, n)` |
# MAGIC | Days between | `DATEDIFF(end, start)` | `datediff(end, start)` |
# MAGIC | Format | `DATE_FORMAT(date, 'pattern')` | `date_format(col, 'pattern')` |
# MAGIC
# MAGIC ### Window Functions
# MAGIC
# MAGIC | Function | SQL | Python |
# MAGIC |----------|-----|--------|
# MAGIC | Row number | `ROW_NUMBER() OVER (...)` | `row_number().over(window)` |
# MAGIC | Rank (gaps) | `RANK() OVER (...)` | `rank().over(window)` |
# MAGIC | Dense rank | `DENSE_RANK() OVER (...)` | `dense_rank().over(window)` |
# MAGIC | Buckets | `NTILE(n) OVER (...)` | `ntile(n).over(window)` |
# MAGIC | Previous row | `LAG(col, n) OVER (...)` | `lag(col, n).over(window)` |
# MAGIC | Next row | `LEAD(col, n) OVER (...)` | `lead(col, n).over(window)` |
# MAGIC | First value | `FIRST_VALUE(col) OVER (...)` | `first(col).over(window)` |
# MAGIC | Running sum | `SUM(col) OVER (... ROWS ...)` | `sum(col).over(window)` |
# MAGIC | Running avg | `AVG(col) OVER (... ROWS ...)` | `avg(col).over(window)` |
# MAGIC
# MAGIC ### Window Frame Syntax
# MAGIC
# MAGIC **SQL:**
# MAGIC ```sql
# MAGIC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW  -- running total
# MAGIC ROWS BETWEEN 6 PRECEDING AND CURRENT ROW          -- 7-day window
# MAGIC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING  -- entire partition
# MAGIC ```
# MAGIC
# MAGIC **Python:**
# MAGIC ```python
# MAGIC Window.partitionBy("col").orderBy("col").rowsBetween(Window.unboundedPreceding, Window.currentRow)
# MAGIC Window.partitionBy("col").orderBy("col").rowsBetween(-6, 0)  # 7-day window
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Week 4 Checklist
# MAGIC
# MAGIC - [ ] Used date extraction functions (YEAR, MONTH, DAYOFWEEK)
# MAGIC - [ ] Applied DATE_TRUNC for time period grouping
# MAGIC - [ ] Performed time-based aggregations (by month, hour, weekday/weekend)
# MAGIC - [ ] Used ranking functions (ROW_NUMBER, RANK, DENSE_RANK, NTILE)
# MAGIC - [ ] Applied LAG/LEAD for period-over-period comparisons
# MAGIC - [ ] Created running totals and moving averages
# MAGIC - [ ] Combined date functions with window functions for time-series analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## Homework Assignment
# MAGIC
# MAGIC 1. **Find the busiest day of each month using window functions**
# MAGIC    - Group flights by month and day
# MAGIC    - Count flights per day
# MAGIC    - Use ROW_NUMBER() to rank days within each month
# MAGIC    - Return only the busiest day per month
# MAGIC
# MAGIC 2. **Calculate week-over-week change in average delays**
# MAGIC    - Aggregate average delays by week (use WEEKOFYEAR or DATE_TRUNC)
# MAGIC    - Use LAG to get the previous week's average
# MAGIC    - Calculate both absolute and percentage change
# MAGIC
# MAGIC 3. **Rank airlines by on-time performance for each month**
# MAGIC    - Define "on-time" as arr_delay <= 0
# MAGIC    - Calculate the on-time percentage for each carrier per month
# MAGIC    - Rank carriers within each month by on-time percentage
# MAGIC
# MAGIC 4. **Identify flights that were worse than the carrier's monthly average**
# MAGIC    - Use a window function to calculate each carrier's monthly average delay
# MAGIC    - Find flights where arr_delay exceeds the carrier's monthly average
# MAGIC    - Return the top 20 worst offenders relative to their carrier's average

