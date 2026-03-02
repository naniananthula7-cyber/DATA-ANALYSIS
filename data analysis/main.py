# Exploratory Data Analysis: E-commerce Sales

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('sales_data.csv')
print(df.head())

# Step 1: Dataset Overview
print("Dataset Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nBasic Statistics:\n", df.describe())

# Convert order_date to datetime
df['order_date'] = pd.to_datetime(df['order_date'])
print("\nDate Range:", df['order_date'].min(), "to", df['order_date'].max())
print(f"Total Revenue: ${df['order_amount'].sum():,.2f}")

# Step 2: Sales by Category
category_sales = df.groupby('category').agg(
    total_revenue=('order_amount', 'sum'),
    avg_order_value=('order_amount', 'mean'),
    order_count=('order_amount', 'count')
).round(2)

category_sales['revenue_pct'] = (category_sales['total_revenue'] /
                                 category_sales['total_revenue'].sum() * 100).round(2)

print(category_sales.sort_values('total_revenue', ascending=False))

category_sales['total_revenue'].plot(kind='bar', figsize=(10, 6), color='steelblue')
plt.title("Revenue by Category")
plt.ylabel("Revenue ($)")
plt.show()

# Step 3: Monthly Sales Trends
df_monthly = df.set_index('order_date').resample('ME').agg({

    'order_amount': ['sum', 'mean', 'count'],
    'customer_id': 'nunique'
})

df_monthly.columns = ['revenue', 'avg_order_value', 'orders', 'active_customers']
print(df_monthly)

df_monthly['revenue'].plot(figsize=(12, 6), marker='o', color='green')
plt.title("Monthly Revenue Trend")
plt.ylabel("Revenue ($)")
plt.xlabel("Month")
plt.grid(True)
plt.show()

# Step 4: Customer Segmentation (RFM Analysis)
current_date = df['order_date'].max()

rfm = df.groupby('customer_id').agg({
    'order_date': lambda x: (current_date - x.max()).days,
    'order_amount': ['count', 'sum', 'mean']
})

rfm.columns = ['recency', 'frequency', 'monetary', 'avg_purchase']

def segment_customer(row):
    if row['recency'] <= 30 and row['frequency'] >= 5:
        return 'VIP'
    elif row['recency'] <= 60 and row['frequency'] >= 3:
        return 'Loyal'
    elif row['recency'] <= 90:
        return 'Active'
    else:
        return 'At Risk'

rfm['segment'] = rfm.apply(segment_customer, axis=1)

segment_summary = rfm.groupby('segment').agg({
    'monetary': ['count', 'mean']
}).round(2)

print(segment_summary)

# Step 5: Top Products Analysis
top_products = df.groupby(['product_id', 'product_name']).agg(
    total_revenue=('order_amount', 'sum'),
    avg_price=('order_amount', 'mean'),
    units_sold=('order_amount', 'count')
).round(2)

top_products['revenue_rank'] = top_products['total_revenue'].rank(ascending=False)
top_products = top_products.sort_values('total_revenue', ascending=False).head(10)

print(top_products)

top_products['total_revenue'].plot(kind='barh', figsize=(10, 6), color='coral')
plt.title("Top 10 Products by Revenue")
plt.xlabel("Revenue ($)")
plt.show()

# Step 6: Geographic Analysis
geo_analysis = df.groupby('region').agg(
    customers=('customer_id', 'nunique'),
    orders=('order_amount', 'count'),
    revenue=('order_amount', 'sum'),
    avg_order_value=('order_amount', 'mean')
).round(2)

geo_analysis['market_share'] = (geo_analysis['revenue'] /
                                geo_analysis['revenue'].sum() * 100).round(2)

print(geo_analysis.sort_values('revenue', ascending=False))

# Step 7: Summary Dashboard (Growth Metrics)
current_month = df[df['order_date'] >= (current_date - pd.DateOffset(months=1))]
previous_month = df[(df['order_date'] < (current_date - pd.DateOffset(months=1))) &
                    (df['order_date'] >= (current_date - pd.DateOffset(months=2)))]

summary = {
    'current_orders': len(current_month),
    'previous_orders': len(previous_month),
    'current_revenue': current_month['order_amount'].sum(),
    'previous_revenue': previous_month['order_amount'].sum(),
    'current_customers': current_month['customer_id'].nunique(),
    'previous_customers': previous_month['customer_id'].nunique()
}

summary['order_growth_pct'] = round(
    (summary['current_orders'] - summary['previous_orders']) /
    summary['previous_orders'] * 100, 2
) if summary['previous_orders'] > 0 else np.nan

summary['revenue_growth_pct'] = round(
    (summary['current_revenue'] - summary['previous_revenue']) /
    summary['previous_revenue'] * 100, 2
) if summary['previous_revenue'] > 0 else np.nan

summary['revenue_per_customer'] = round(
    summary['current_revenue'] / summary['current_customers'], 2
) if summary['current_customers'] > 0 else np.nan


print("\nGrowth Metrics:")
for key, value in summary.items():
    print(f"{key}: {value}")
