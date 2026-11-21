# Retail Sales Performance Analysis
# Student: Devesh Mohan Gupta
# Admission No.: 2026B15410191
# Program: Data Analytics
#
# Beginner-level project using Python, NumPy and Pandas.
# Keep this .py file in the same folder as: 01_shop_ sales.csv
#
# IMPORTANT:
# First run the program. It prints the real column names in your CSV.
# If your column names differ, update the column names in the
# "COLUMN NAME SETTINGS" section below.

import pandas as pd
import numpy as np

# -------------------- 1. LOAD DATASET --------------------

file_name = "01_shop_ sales.csv"

data = pd.read_csv(file_name)

print("RETAIL SALES PERFORMANCE ANALYSIS")
print("----------------------------------")

# -------------------- 2. INSPECT DATA --------------------

print("\nFirst five rows:")
print(data.head())

print("\nDataset shape (rows, columns):")
print(data.shape)

print("\nColumn names:")
print(data.columns.tolist())

print("\nData types:")
print(data.dtypes)

print("\nMissing values in each column:")
print(data.isnull().sum())

print("\nNumber of completely duplicate rows:")
print(data.duplicated().sum())

# -------------------- 3. COLUMN NAME SETTINGS --------------------
# Change the text on the right if your CSV uses different column names.
# These are expected names from the project brief, not guaranteed names.

ORDER_ID = "Order_ID"
ORDER_DATE = "Order_Date"
PRODUCT = "Product"
CATEGORY = "Category"
QUANTITY = "Quantity"
UNIT_PRICE = "Unit_Price"
REVENUE = "Revenue"
CITY = "City"
CUSTOMER_ID = "Customer_ID"
PAYMENT_METHOD = "Payment_Method"

# Check which expected columns are available.
required_for_revenue = [QUANTITY, UNIT_PRICE]
missing_columns = [col for col in required_for_revenue if col not in data.columns]

if missing_columns:
    print("\nThe following columns needed to calculate revenue were not found:")
    print(missing_columns)
    print("Please update the column name settings after checking the printed column list.")
else:
    # -------------------- 4. CLEAN DATA --------------------

    if ORDER_ID in data.columns:
        print("\nDuplicate order IDs:", data[ORDER_ID].duplicated().sum())

    if CITY in data.columns:
        print("\nMissing city values before cleaning:", data[CITY].isnull().sum())
        data[CITY] = data[CITY].fillna("Unknown")
        print("Missing city values after cleaning:", data[CITY].isnull().sum())

    if ORDER_DATE in data.columns:
        data[ORDER_DATE] = pd.to_datetime(data[ORDER_DATE], errors="coerce")
        data["Month"] = data[ORDER_DATE].dt.month_name()
        data["Weekday"] = data[ORDER_DATE].dt.day_name()

    # Convert numeric columns safely.
    data[QUANTITY] = pd.to_numeric(data[QUANTITY], errors="coerce")
    data[UNIT_PRICE] = pd.to_numeric(data[UNIT_PRICE], errors="coerce")

    # Calculate revenue from quantity and unit price.
    data["Calculated_Revenue"] = data[QUANTITY] * data[UNIT_PRICE]

    print("\nFirst calculated revenue values:")
    print(data[["Calculated_Revenue"]].head())

    if REVENUE in data.columns:
        data[REVENUE] = pd.to_numeric(data[REVENUE], errors="coerce")
        data["Revenue_Difference"] = data["Calculated_Revenue"] - data[REVENUE]
        print("\nRevenue verification (first five rows):")
        print(data[["Calculated_Revenue", REVENUE, "Revenue_Difference"]].head())
        # Use calculated revenue for analysis so the calculation is transparent.
        data["Analysis_Revenue"] = data["Calculated_Revenue"]
    else:
        data["Analysis_Revenue"] = data["Calculated_Revenue"]

    # Remove rows where revenue cannot be calculated.
    data = data.dropna(subset=["Analysis_Revenue"])

    # -------------------- 5. BASIC SALES SUMMARY --------------------

    total_revenue = data["Analysis_Revenue"].sum()
    total_quantity = data[QUANTITY].sum()

    print("\nSALES SUMMARY")
    print("Total Revenue:", round(total_revenue, 2))
    print("Total Quantity Sold:", total_quantity)

    # If one row represents one order, average order value is the mean row revenue.
    # If there are multiple rows per order, aggregate by Order_ID first.
    if ORDER_ID in data.columns:
        order_totals = data.groupby(ORDER_ID)["Analysis_Revenue"].sum()
        average_order_value = order_totals.mean()
        total_orders = order_totals.count()
    else:
        average_order_value = data["Analysis_Revenue"].mean()
        total_orders = len(data)

    print("Total Orders:", total_orders)
    print("Average Order Value:", round(average_order_value, 2))

    # -------------------- 6. TOP 10 PRODUCTS --------------------

    if PRODUCT in data.columns:
        print("\nTOP 10 PRODUCTS BY REVENUE")
        product_sales = data.groupby(PRODUCT)["Analysis_Revenue"].sum()
        print(product_sales.sort_values(ascending=False).head(10))

    # -------------------- 7. CITY-WISE REVENUE --------------------

    if CITY in data.columns:
        print("\nCITY-WISE REVENUE")
        city_sales = data.groupby(CITY)["Analysis_Revenue"].sum()
        print(city_sales.sort_values(ascending=False))

    # -------------------- 8. MONTHLY REVENUE --------------------

    if "Month" in data.columns:
        print("\nMONTHLY REVENUE")
        monthly_sales = data.groupby("Month")["Analysis_Revenue"].sum()
        print(monthly_sales.sort_values(ascending=False))
        if len(monthly_sales) > 0:
            print("Best month:", monthly_sales.idxmax())

    # -------------------- 9. TOP 10 CUSTOMERS --------------------

    if CUSTOMER_ID in data.columns:
        print("\nTOP 10 CUSTOMERS BY TOTAL SPENDING")
        customer_sales = data.groupby(CUSTOMER_ID)["Analysis_Revenue"].sum()
        print(customer_sales.sort_values(ascending=False).head(10))

    # -------------------- 10. PAYMENT METHOD COMPARISON --------------------

    if PAYMENT_METHOD in data.columns:
        print("\nPAYMENT METHOD SUMMARY")
        if ORDER_ID in data.columns:
            payment_summary = data.groupby(PAYMENT_METHOD).agg(
                Orders=(ORDER_ID, "nunique"),
                Revenue=("Analysis_Revenue", "sum")
            )
        else:
            payment_summary = data.groupby(PAYMENT_METHOD).agg(
                Orders=("Analysis_Revenue", "count"),
                Revenue=("Analysis_Revenue", "sum")
            )
        print(payment_summary.sort_values("Revenue", ascending=False))

    # -------------------- 11. PRODUCT CATEGORY SUMMARY --------------------

    if CATEGORY in data.columns:
        print("\nPRODUCT CATEGORY SUMMARY")
        category_summary = data.groupby(CATEGORY).agg(
            Total_Quantity=(QUANTITY, "sum"),
            Total_Revenue=("Analysis_Revenue", "sum")
        )
        print(category_summary.sort_values("Total_Revenue", ascending=False))

    # -------------------- 12. NUMPY STATISTICS --------------------

    revenue_array = np.array(data["Analysis_Revenue"])

    print("\nNUMPY REVENUE STATISTICS")
    print("Mean:", round(np.mean(revenue_array), 2))
    print("Median:", round(np.median(revenue_array), 2))
    print("Standard Deviation:", round(np.std(revenue_array), 2))

    # -------------------- 13. IQR OUTLIER CHECK --------------------

    q1 = np.percentile(revenue_array, 25)
    q3 = np.percentile(revenue_array, 75)
    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    outliers = data[
        (data["Analysis_Revenue"] < lower_limit) |
        (data["Analysis_Revenue"] > upper_limit)
    ]

    print("\nIQR OUTLIER ANALYSIS")
    print("Q1:", round(q1, 2))
    print("Q3:", round(q3, 2))
    print("IQR:", round(iqr, 2))
    print("Lower limit:", round(lower_limit, 2))
    print("Upper limit:", round(upper_limit, 2))
    print("Number of unusual revenue records:", len(outliers))
    print(outliers.head(10))

    # -------------------- 14. SAVE CLEANED DATA --------------------

    data.to_csv("cleaned_retail_sales.csv", index=False)
    print("\nCleaned data saved as cleaned_retail_sales.csv")

print("\nPROJECT EXECUTION FINISHED")
