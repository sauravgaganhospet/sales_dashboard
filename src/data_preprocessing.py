import pandas as pd


def load_sales_data(filepath):
    df = pd.read_csv(filepath)

    # Remove duplicates
    df = df.drop_duplicates()

    # Standardize date format
    df['date'] = pd.to_datetime(df['date'])

    # Handle missing values
    df = df.fillna({
        'units_sold': 0,
        'unit_price': 0
    })

    # Feature Engineering
    df['revenue'] = df['units_sold'] * df['unit_price']
    df['month'] = df['date'].dt.to_period('M')

    return df


def calculate_kpis(df):
    total_revenue = df['revenue'].sum()
    total_orders = df.shape[0]
    avg_order_value = total_revenue / total_orders

    revenue_by_category = df.groupby('category')['revenue'].sum()
    revenue_by_region = df.groupby('region')['revenue'].sum()

    monthly_revenue = df.groupby('month')['revenue'].sum().sort_index()
    mom_growth = monthly_revenue.pct_change().fillna(0)

    return {
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "avg_order_value": avg_order_value,
        "revenue_by_category": revenue_by_category,
        "revenue_by_region": revenue_by_region,
        "monthly_revenue": monthly_revenue,
        "mom_growth": mom_growth
    }
