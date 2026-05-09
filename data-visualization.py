# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Week 5: Data Visualization
# MAGIC
# MAGIC Visualization turns numbers into stories. This week you'll learn to create charts using three Python
# MAGIC libraries — **matplotlib** for fundamentals, **seaborn** for statistical graphics, and **plotly** for
# MAGIC interactive exploration. SQL stays in the picture for data preparation, feeding clean aggregations
# MAGIC into your plots.

# COMMAND ----------

# Imports for this notebook
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: The Visualization Workflow
# MAGIC
# MAGIC Databricks notebooks support two approaches to visualization:
# MAGIC
# MAGIC 1. **Click-to-chart**: Run `display(df)` and click the chart icon below the results. Quick for
# MAGIC exploration, but limited customization and not reproducible.
# MAGIC 2. **Programmatic plotting**: Write Python code to create exactly the chart you want. Reproducible,
# MAGIC customizable, and shareable.
# MAGIC
# MAGIC This week focuses on programmatic plotting. The workflow is always the same:
# MAGIC
# MAGIC ```python
# MAGIC # Step 1: Prepare data with SQL or PySpark
# MAGIC result = spark.sql("SELECT carrier, COUNT(*) AS flights FROM flights GROUP BY carrier")
# MAGIC
# MAGIC # Step 2: Convert to pandas (required for plotting libraries)
# MAGIC result_pd = result.toPandas()
# MAGIC
# MAGIC # Step 3: Plot
# MAGIC plt.bar(result_pd['carrier'], result_pd['flights'])
# MAGIC plt.show()
# MAGIC ```
# MAGIC
# MAGIC The `.toPandas()` call collects all data to the driver node, so **aggregate or sample first** when
# MAGIC working with large datasets.
# MAGIC
# MAGIC ### Choosing the Right Chart Type
# MAGIC
# MAGIC | Question | Chart Type | Example |
# MAGIC |----------|-----------|---------|
# MAGIC | How much of each category? | Bar chart | Flights per carrier |
# MAGIC | How is a value distributed? | Histogram | Diamond price distribution |
# MAGIC | How does a value change over time? | Line chart | Average delay by month |
# MAGIC | How do two variables relate? | Scatter plot | Carat vs price |
# MAGIC | How do distributions compare across groups? | Box / Violin plot | Delay by carrier |
# MAGIC | How do three variables interact? | Heatmap | Delay by carrier × month |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Matplotlib Fundamentals
# MAGIC
# MAGIC Matplotlib is Python's foundational plotting library. We imported it above as `plt`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bar Chart — Flights per Carrier
# MAGIC
# MAGIC Bar charts compare quantities across categories.

# COMMAND ----------

carrier_counts = spark.sql("""
    SELECT carrier, COUNT(*) AS flight_count
    FROM flights
    GROUP BY carrier
    ORDER BY flight_count DESC
""").toPandas()

plt.figure(figsize=(10, 6))
plt.bar(carrier_counts['carrier'], carrier_counts['flight_count'], color='steelblue')
plt.xlabel('Carrier')
plt.ylabel('Number of Flights')
plt.title('Flights per Carrier')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Histogram — Diamond Price Distribution
# MAGIC
# MAGIC Histograms show how a single variable is distributed. The `bins` parameter controls how many
# MAGIC bars to split the data into. `alpha` controls transparency (0 = invisible, 1 = solid).

# COMMAND ----------

# Load diamonds as pandas — we'll reuse this throughout the notebook
diamonds_pd = spark.table("diamonds").toPandas()

plt.figure(figsize=(10, 6))
plt.hist(diamonds_pd['price'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Price ($)')
plt.ylabel('Count')
plt.title('Distribution of Diamond Prices')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Line Chart — Average Delay by Month
# MAGIC
# MAGIC Line charts show trends over an ordered variable like time. `marker='o'` adds dots at each
# MAGIC data point. `plt.grid(True, alpha=0.3)` adds subtle gridlines.

# COMMAND ----------

monthly_delay = spark.sql("""
    SELECT month, AVG(arr_delay) AS avg_delay
    FROM flights
    WHERE arr_delay IS NOT NULL
    GROUP BY month
    ORDER BY month
""").toPandas()

plt.figure(figsize=(10, 6))
plt.plot(monthly_delay['month'], monthly_delay['avg_delay'], marker='o', color='steelblue')
plt.xlabel('Month')
plt.ylabel('Average Arrival Delay (minutes)')
plt.title('Average Flight Delay by Month')
plt.xticks(range(1, 13))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scatter Plot — Carat vs Price by Cut
# MAGIC
# MAGIC Scatter plots reveal relationships between two numeric variables. In Week 2 we made a basic
# MAGIC scatter of iris data. Here's a more complete version — with color encoding and low `alpha` to
# MAGIC handle 53,000 overlapping points:

# COMMAND ----------

plt.figure(figsize=(10, 6))
for cut in ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair']:
    subset = diamonds_pd[diamonds_pd['cut'] == cut]
    plt.scatter(subset['carat'], subset['price'], alpha=0.1, s=10, label=cut)

plt.xlabel('Carat')
plt.ylabel('Price ($)')
plt.title('Diamond Price vs Carat by Cut')
plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Customizing Charts
# MAGIC
# MAGIC ### Labels, Titles, and Formatting
# MAGIC
# MAGIC Every chart should have clear labels. Common customizations:

# COMMAND ----------

plt.figure(figsize=(10, 6))
plt.bar(carrier_counts['carrier'], carrier_counts['flight_count'], color='steelblue')

# Text
plt.xlabel('Carrier Code', fontsize=12)
plt.ylabel('Number of Flights', fontsize=12)
plt.title('Flights per Carrier in 2013', fontsize=14)

# Grid
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Color choices**: Use named colors (`'steelblue'`, `'coral'`, `'forestgreen'`) for single-series
# MAGIC charts. Use color to encode a variable (like cut or carrier) only when it adds information.
# MAGIC
# MAGIC ### Subplots — Side by Side Comparison
# MAGIC
# MAGIC `plt.subplots()` creates multiple charts in one figure. With subplots, use `ax.set_xlabel()`
# MAGIC instead of `plt.xlabel()` — each subplot has its own methods.

# COMMAND ----------

monthly_delays = spark.sql("""
    SELECT month,
           AVG(dep_delay) AS avg_dep_delay,
           AVG(arr_delay) AS avg_arr_delay
    FROM flights
    WHERE dep_delay IS NOT NULL AND arr_delay IS NOT NULL
    GROUP BY month
    ORDER BY month
""").toPandas()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(monthly_delays['month'], monthly_delays['avg_dep_delay'], marker='o', color='steelblue')
ax1.set_title('Average Departure Delay')
ax1.set_xlabel('Month')
ax1.set_ylabel('Delay (minutes)')
ax1.set_xticks(range(1, 13))
ax1.grid(True, alpha=0.3)

ax2.plot(monthly_delays['month'], monthly_delays['avg_arr_delay'], marker='s', color='coral')
ax2.set_title('Average Arrival Delay')
ax2.set_xlabel('Month')
ax2.set_ylabel('Delay (minutes)')
ax2.set_xticks(range(1, 13))
ax2.grid(True, alpha=0.3)

fig.suptitle('Flight Delays by Month', fontsize=14)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Seaborn for Statistical Visualization
# MAGIC
# MAGIC Seaborn builds on matplotlib with better defaults and built-in statistical calculations.
# MAGIC We imported it above as `sns`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Box Plot — Delay Distribution by Carrier
# MAGIC
# MAGIC Box plots show the median, quartiles, and outliers. Filter extreme values for cleaner charts —
# MAGIC you're not hiding data, you're focusing the view.

# COMMAND ----------

delay_data = spark.sql("""
    SELECT carrier, arr_delay
    FROM flights
    WHERE arr_delay BETWEEN -60 AND 120
""").toPandas()

plt.figure(figsize=(12, 6))
sns.boxplot(x='carrier', y='arr_delay', data=delay_data)
plt.title('Arrival Delay Distribution by Carrier')
plt.xlabel('Carrier')
plt.ylabel('Arrival Delay (minutes)')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Violin Plot — Diamond Price by Cut
# MAGIC
# MAGIC Violin plots show the full distribution shape. Wider sections indicate where more data
# MAGIC points fall.

# COMMAND ----------

plt.figure(figsize=(10, 6))
sns.violinplot(x='cut', y='price', data=diamonds_pd,
               order=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
plt.title('Diamond Price Distribution by Cut')
plt.xlabel('Cut')
plt.ylabel('Price ($)')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Heatmap — Average Delay by Carrier and Month
# MAGIC
# MAGIC Heatmaps display a matrix of values with color intensity. `annot=True` prints values in each
# MAGIC cell. `cmap='RdYlGn_r'` uses red for high delays and green for low. `center=0` puts the
# MAGIC neutral color at zero delay.

# COMMAND ----------

heatmap_data = spark.sql("""
    SELECT a.name AS airline, f.month,
           ROUND(AVG(f.arr_delay), 1) AS avg_delay
    FROM flights f
    JOIN airlines a ON f.carrier = a.carrier
    WHERE f.arr_delay IS NOT NULL
    GROUP BY a.name, f.month
""").toPandas()

pivot = heatmap_data.pivot_table(index='airline', columns='month', values='avg_delay')

plt.figure(figsize=(14, 8))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn_r', center=0)
plt.title('Average Arrival Delay by Airline and Month (minutes)')
plt.xlabel('Month')
plt.ylabel('')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Count Plot — Diamond Cut Distribution
# MAGIC
# MAGIC Count plots are bar charts that automatically count occurrences:

# COMMAND ----------

plt.figure(figsize=(10, 6))
sns.countplot(x='cut', data=diamonds_pd,
              order=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
plt.title('Distribution of Diamond Cuts')
plt.xlabel('Cut')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pair Plot — Iris Multi-Variable Relationships
# MAGIC
# MAGIC Pair plots create scatter plots for every combination of numeric variables — useful for
# MAGIC initial exploration. This creates a 4×4 grid: scatter plots off the diagonal, distributions
# MAGIC on the diagonal, colored by species.

# COMMAND ----------

iris_pd = spark.table("iris").toPandas()

sns.pairplot(iris_pd, hue='Species', height=2.5)
plt.suptitle('Iris Dataset: Pairwise Relationships', y=1.02)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5: Plotly for Interactive Charts
# MAGIC
# MAGIC Plotly creates charts you can hover over, zoom into, and filter. We imported it above as `px`.
# MAGIC
# MAGIC **Performance tip**: Plotly sends all data to the browser. Sample large datasets first:

# COMMAND ----------

# Sample 10% of diamonds for plotly charts
diamonds_sample = spark.table("diamonds").sample(fraction=0.1).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interactive Scatter — Carat vs Price
# MAGIC
# MAGIC Hover over any point to see its details. Click legend entries to toggle groups on and off.

# COMMAND ----------

fig = px.scatter(diamonds_sample, x='carat', y='price', color='cut',
                 hover_data=['clarity', 'color'],
                 title='Diamond Price vs Carat (hover for details)',
                 opacity=0.5)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interactive Bar — Flights per Carrier
# MAGIC
# MAGIC Color encodes a second variable (average delay) on a continuous scale.

# COMMAND ----------

carrier_stats = spark.sql("""
    SELECT a.name, COUNT(*) AS flights,
           ROUND(AVG(f.arr_delay), 1) AS avg_delay
    FROM flights f
    JOIN airlines a ON f.carrier = a.carrier
    WHERE f.arr_delay IS NOT NULL
    GROUP BY a.name
    ORDER BY flights DESC
""").toPandas()

fig = px.bar(carrier_stats, x='name', y='flights', color='avg_delay',
             title='Flights per Carrier (colored by average delay)',
             color_continuous_scale='RdYlGn_r')
fig.update_layout(xaxis_tickangle=-45)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interactive Line — Monthly Delay Trends by Carrier
# MAGIC
# MAGIC Click carrier names in the legend to toggle them — useful for comparing specific airlines.

# COMMAND ----------

monthly_by_carrier = spark.sql("""
    SELECT carrier, month, ROUND(AVG(arr_delay), 1) AS avg_delay
    FROM flights
    WHERE arr_delay IS NOT NULL
    GROUP BY carrier, month
    ORDER BY carrier, month
""").toPandas()

fig = px.line(monthly_by_carrier, x='month', y='avg_delay', color='carrier',
              title='Monthly Delay Trends by Carrier',
              markers=True)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interactive Histogram — Price Distribution by Cut
# MAGIC
# MAGIC Try changing `barmode` to `'stack'` or `'group'` to see different views of the same data.

# COMMAND ----------

fig = px.histogram(diamonds_sample, x='price', color='cut',
                   title='Diamond Price Distribution by Cut',
                   barmode='overlay', opacity=0.6, nbins=50)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 6: Combining SQL Data Prep with Visualization
# MAGIC
# MAGIC These examples show the complete workflow from SQL query to finished chart, tying together
# MAGIC skills from Weeks 3–5.

# COMMAND ----------

# MAGIC %md
# MAGIC ### On-Time Performance by Airline
# MAGIC
# MAGIC Horizontal bar charts work well when category labels are long. `ORDER BY ontime_pct` in SQL
# MAGIC ensures the bars are sorted.

# COMMAND ----------

ontime = spark.sql("""
    SELECT a.name,
           COUNT(*) AS total_flights,
           ROUND(100.0 * SUM(CASE WHEN f.arr_delay <= 0 THEN 1 ELSE 0 END)
                 / COUNT(*), 1) AS ontime_pct
    FROM flights f
    JOIN airlines a ON f.carrier = a.carrier
    WHERE f.arr_delay IS NOT NULL
    GROUP BY a.name
    ORDER BY ontime_pct
""").toPandas()

plt.figure(figsize=(10, 8))
plt.barh(ontime['name'], ontime['ontime_pct'], color='steelblue')
plt.xlabel('On-Time Percentage (%)')
plt.title('Airline On-Time Arrival Performance')
plt.xlim(0, 100)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Price per Carat by Cut with Error Bars
# MAGIC
# MAGIC Error bars show variability within each group. SQL computes the mean and standard deviation,
# MAGIC and matplotlib draws the bars.

# COMMAND ----------

price_stats = spark.sql("""
    SELECT cut,
           AVG(price / carat) AS avg_ppc,
           STDDEV(price / carat) AS std_ppc
    FROM diamonds
    GROUP BY cut
""").toPandas()

cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
price_stats['cut'] = pd.Categorical(price_stats['cut'], categories=cut_order, ordered=True)
price_stats = price_stats.sort_values('cut')

plt.figure(figsize=(10, 6))
plt.bar(price_stats['cut'].astype(str), price_stats['avg_ppc'],
        yerr=price_stats['std_ppc'], capsize=5, color='steelblue', alpha=0.8)
plt.xlabel('Cut')
plt.ylabel('Price per Carat ($)')
plt.title('Average Price per Carat by Cut (± std dev)')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick Reference Tables
# MAGIC
# MAGIC ### Matplotlib Basics
# MAGIC
# MAGIC | Operation | Code |
# MAGIC |-----------|------|
# MAGIC | Create figure | `plt.figure(figsize=(10, 6))` |
# MAGIC | Bar chart | `plt.bar(x, y, color='steelblue')` |
# MAGIC | Horizontal bar | `plt.barh(y, x)` |
# MAGIC | Line chart | `plt.plot(x, y, marker='o')` |
# MAGIC | Histogram | `plt.hist(data, bins=50, alpha=0.7)` |
# MAGIC | Scatter plot | `plt.scatter(x, y, alpha=0.5, s=10)` |
# MAGIC | Subplots | `fig, (ax1, ax2) = plt.subplots(1, 2)` |
# MAGIC | Labels | `plt.xlabel('X')`, `plt.ylabel('Y')` |
# MAGIC | Title | `plt.title('Title', fontsize=14)` |
# MAGIC | Legend | `plt.legend()` |
# MAGIC | Grid | `plt.grid(True, alpha=0.3)` |
# MAGIC | Show | `plt.tight_layout()` then `plt.show()` |
# MAGIC
# MAGIC ### Seaborn
# MAGIC
# MAGIC | Chart Type | Code |
# MAGIC |-----------|------|
# MAGIC | Box plot | `sns.boxplot(x='group', y='value', data=df)` |
# MAGIC | Violin plot | `sns.violinplot(x='group', y='value', data=df)` |
# MAGIC | Heatmap | `sns.heatmap(pivot_df, annot=True, cmap='RdYlGn_r')` |
# MAGIC | Count plot | `sns.countplot(x='category', data=df)` |
# MAGIC | Pair plot | `sns.pairplot(df, hue='group')` |
# MAGIC
# MAGIC ### Plotly Express
# MAGIC
# MAGIC | Chart Type | Code |
# MAGIC |-----------|------|
# MAGIC | Scatter | `px.scatter(df, x='x', y='y', color='group')` |
# MAGIC | Bar | `px.bar(df, x='x', y='y', color='value')` |
# MAGIC | Line | `px.line(df, x='x', y='y', color='group')` |
# MAGIC | Histogram | `px.histogram(df, x='x', color='group')` |
# MAGIC
# MAGIC ### Data Preparation Workflow
# MAGIC
# MAGIC | Step | Code |
# MAGIC |------|------|
# MAGIC | SQL to pandas | `spark.sql("SELECT ...").toPandas()` |
# MAGIC | Table to pandas | `spark.table("name").toPandas()` |
# MAGIC | Sample large data | `spark.table("name").sample(fraction=0.1).toPandas()` |
# MAGIC | Pivot for heatmap | `df.pivot_table(index='row', columns='col', values='val')` |
# MAGIC | Categorical order | `pd.Categorical(df['col'], categories=[...], ordered=True)` |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Week 5 Checklist
# MAGIC
# MAGIC - [ ] Created a bar chart with matplotlib
# MAGIC - [ ] Created a histogram showing a distribution
# MAGIC - [ ] Created a line chart showing a trend over time
# MAGIC - [ ] Created a scatter plot with color encoding
# MAGIC - [ ] Created a seaborn box plot or violin plot
# MAGIC - [ ] Created a seaborn heatmap
# MAGIC - [ ] Created an interactive plotly chart
# MAGIC - [ ] Built a chart from a SQL aggregation (full workflow)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Homework Assignment
# MAGIC
# MAGIC 1. **Chart Type Comparison**
# MAGIC    - Pick one question about the diamonds dataset (e.g., "How does price vary by cut?")
# MAGIC    - Answer it with three different chart types (e.g., bar, box plot, violin)
# MAGIC    - Write one sentence about what each chart type reveals that the others don't
# MAGIC
# MAGIC 2. **Flight Delay Dashboard**
# MAGIC    - Create a 2×2 subplot figure (`plt.subplots(2, 2)`) with four charts:
# MAGIC      - Top left: bar chart of flights per carrier
# MAGIC      - Top right: line chart of average delay by month
# MAGIC      - Bottom left: histogram of departure delays (filter to -60 to 120 minutes)
# MAGIC      - Bottom right: box plot of arrival delay by carrier (top 5 carriers only)
# MAGIC    - Add a main title with `fig.suptitle()`
# MAGIC
# MAGIC 3. **Interactive Exploration**
# MAGIC    - Create a plotly scatter plot of diamonds with at least 3 variables mapped (x, y, color, size, or hover)
# MAGIC    - Use it to find one interesting pattern and write 2-3 sentences describing what you found
# MAGIC
# MAGIC 4. **Share Your Best Chart**
# MAGIC    - Pick your favorite chart from this week's work
# MAGIC    - Post a screenshot to Slack with a one-sentence caption explaining what the chart shows

