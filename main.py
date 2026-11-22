
import csv
import os
import numpy as np

# ------------------------------------------------------------
# 0. SETTINGS AND BUSINESS CODE DICTIONARIES
# ------------------------------------------------------------

RANDOM_SEED = 42
NUMBER_OF_ROWS = 5000
OUTPUT_FOLDER = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(OUTPUT_FOLDER, "freshflow.csv")

COLUMN_NAMES = [
    "transaction_id", "day_index", "outlet_id", "product_code",
    "category_code", "quantity", "unit_price", "unit_cost",
    "discount_rate", "stock_before_units", "payment_code",
    "customer_rating"
]

# Outlet codes
OUTLET_NAMES = {
    1: "Outlet A",
    2: "Outlet B",
    3: "Outlet C"
}

# Product code -> product name
PRODUCT_NAMES = {
    1: "Orange Juice",
    2: "Mango Juice",
    3: "Apple Juice",
    4: "Watermelon Juice",
    5: "Banana Smoothie",
    6: "Mixed Fruit Smoothie",
    7: "Lemonade",
    8: "Iced Tea"
}

# Product code -> correct category code
PRODUCT_CATEGORY = {
    1: 1, 2: 1, 3: 1, 4: 1,
    5: 2, 6: 2,
    7: 3,
    8: 4
}

CATEGORY_NAMES = {
    1: "Juice",
    2: "Smoothie",
    3: "Lemonade",
    4: "Tea"
}

PAYMENT_NAMES = {
    1: "Cash",
    2: "UPI",
    3: "Card",
    4: "Other digital payment"
}


# ------------------------------------------------------------
# 1. DATA GENERATION: CREATE freshflow.csv
# ------------------------------------------------------------

def generate_freshflow_csv(file_path=CSV_FILE, seed=RANDOM_SEED):
    """
    Creates exactly 5,000 rows and 12 numeric columns.
    Deliberate data-quality cases are injected for the assignment.
    Blank cells represent missing numeric measurements.
    """
    rng = np.random.default_rng(seed)

    # Create a realistic distribution of transaction data.
    transaction_id = np.arange(100001, 100001 + NUMBER_OF_ROWS)
    day_index = rng.integers(1, 31, NUMBER_OF_ROWS)
    outlet_id = rng.choice([1, 2, 3], NUMBER_OF_ROWS, p=[0.38, 0.34, 0.28])
    product_code = rng.integers(1, 9, NUMBER_OF_ROWS)

    category_code = np.array(
        [PRODUCT_CATEGORY[int(product)] for product in product_code]
    )

    quantity = rng.integers(1, 9, NUMBER_OF_ROWS).astype(float)

    # Product-specific normal selling prices and costs (INR).
    price_ranges = {
        1: (65, 100), 2: (70, 110), 3: (60, 95), 4: (65, 105),
        5: (95, 145), 6: (105, 155), 7: (35, 65), 8: (30, 55)
    }
    cost_ranges = {
        1: (25, 42), 2: (28, 48), 3: (24, 40), 4: (27, 45),
        5: (42, 70), 6: (48, 78), 7: (12, 25), 8: (10, 22)
    }

    unit_price = np.zeros(NUMBER_OF_ROWS, dtype=float)
    unit_cost = np.zeros(NUMBER_OF_ROWS, dtype=float)

    for product in range(1, 9):
        mask = product_code == product
        unit_price[mask] = rng.uniform(
            price_ranges[product][0], price_ranges[product][1], mask.sum()
        )
        unit_cost[mask] = rng.uniform(
            cost_ranges[product][0], cost_ranges[product][1], mask.sum()
        )

    unit_price = np.round(unit_price, 2)
    unit_cost = np.round(unit_cost, 2)
    discount_rate = np.round(rng.uniform(0.00, 0.25, NUMBER_OF_ROWS), 3)
    stock_before_units = rng.integers(5, 100, NUMBER_OF_ROWS).astype(float)
    payment_code = rng.integers(1, 5, NUMBER_OF_ROWS)
    customer_rating = rng.integers(1, 6, NUMBER_OF_ROWS).astype(float)

    data = np.column_stack((
        transaction_id, day_index, outlet_id, product_code, category_code,
        quantity, unit_price, unit_cost, discount_rate, stock_before_units,
        payment_code, customer_rating
    )).astype(float)

    # Use different rows for each injected issue so the generator is reproducible.
    available = np.arange(NUMBER_OF_ROWS)
    rng.shuffle(available)
    positions = {}
    cursor = 0

    def take(count):
        nonlocal cursor
        selected = available[cursor:cursor + count]
        cursor += count
        return selected

    # Missing measurements: 4 prices, 4 costs and 20 ratings.
    data[take(4), 6] = np.nan
    data[take(4), 7] = np.nan
    data[take(20), 11] = np.nan

    # Duplicate transaction IDs: five later records reuse an earlier ID.
    duplicate_rows = take(5)
    for row in duplicate_rows:
        earlier_row = int(rng.integers(0, row)) if row > 0 else 1
        data[row, 0] = data[earlier_row, 0]

    # Invalid or suspicious values.
    data[take(4), 5] = rng.choice([0, -1, -2], 4)           # quantity
    data[take(2), 6] = -rng.uniform(10, 50, 2)              # negative price
    data[take(2), 7] = -rng.uniform(5, 25, 2)               # negative cost
    data[take(2), 8] = rng.uniform(1.05, 1.50, 2)           # discount > 1
    data[take(3), 11] = rng.choice([0, 6, 7], 3)             # invalid rating

    # Four product/category mapping mismatches.
    mismatch_rows = take(4)
    for row in mismatch_rows:
        correct = int(data[row, 4])
        possible = [code for code in range(1, 5) if code != correct]
        data[row, 4] = int(rng.choice(possible))

    data[take(2), 2] = rng.choice([0, 4, 9], 2)              # unknown outlet
    data[take(3), 6] = rng.uniform(350, 650, 3)              # high but plausible price
    data[take(5), 8] = rng.uniform(0.35, 0.60, 5)            # high valid discount
    data[take(5), 9] = 0                                    # zero stock before sale

    # Round numeric measurements for a clean CSV.
    data[:, 6:9] = np.round(data[:, 6:9], 3)

    # Write the numeric CSV. Missing values are written as blank cells.
    formats = [
        "%.0f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f",
        "%.3f", "%.3f", "%.3f", "%.0f", "%.0f", "%.0f"
    ]
    with open(file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(COLUMN_NAMES)
        for row in data:
            output_row = []
            for value, fmt in zip(row, formats):
                if np.isnan(value):
                    output_row.append("")
                else:
                    output_row.append(fmt % value)
            writer.writerow(output_row)

    print("\nCSV GENERATION")
    print("---------------")
    print("Created:", file_path)
    print("Data rows:", NUMBER_OF_ROWS)
    print("Columns:", len(COLUMN_NAMES))
    print("Random seed:", seed)
    return file_path


# ------------------------------------------------------------
# 2. LOAD DATA USING NUMPY ONLY
# ------------------------------------------------------------

def load_data(file_path):
    """Load numeric data; blank CSV cells are converted to NaN."""
    data = np.genfromtxt(
        file_path,
        delimiter=",",
        skip_header=1,
        dtype=float,
        ndmin=2,
        filling_values=np.nan
    )
    return data


# ------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ------------------------------------------------------------

def column(data, name):
    """Return one column using its position in COLUMN_NAMES."""
    return data[:, COLUMN_NAMES.index(name)]


def safe_sum(values):
    return float(np.nansum(values))


def safe_mean(values):
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return np.nan
    return float(np.mean(valid))


def safe_percentage(numerator, denominator):
    if not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return float((numerator / denominator) * 100)


def print_section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def save_numeric_report(file_name, header, rows):
    """Save a small numeric report as a CSV file."""
    path = os.path.join(OUTPUT_FOLDER, file_name)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    print("Saved report:", path)


# ------------------------------------------------------------
# 4. PHASE 1 / TASK A: DATA PROFILE
# ------------------------------------------------------------

def data_profile(data):
    print_section("TASK A - DATA PROFILE")

    print("Array shape:", data.shape)
    print("Number of dimensions:", data.ndim)
    print("Data type:", data.dtype)
    print("Total values:", data.size)
    print("Memory used (bytes):", data.nbytes)
    print("Rows loaded:", data.shape[0])
    print("Columns loaded:", data.shape[1])

    print("\nColumn inventory:")
    for index, name in enumerate(COLUMN_NAMES):
        values = data[:, index]
        missing = int(np.isnan(values).sum())
        finite = values[np.isfinite(values)]
        if finite.size:
            print(
                f"{name:22s} min={np.min(finite):10.2f} "
                f"max={np.max(finite):10.2f} missing={missing}"
            )
        else:
            print(f"{name:22s} no valid values; missing={missing}")

    print("\nFirst five rows:")
    print(data[:5])


# ------------------------------------------------------------
# 5. TASK B: DATA INTEGRITY AUDIT
# ------------------------------------------------------------

def integrity_audit(data):
    print_section("TASK B - INTEGRITY AUDIT")

    transaction = column(data, "transaction_id")
    day = column(data, "day_index")
    outlet = column(data, "outlet_id")
    product = column(data, "product_code")
    category = column(data, "category_code")
    quantity = column(data, "quantity")
    price = column(data, "unit_price")
    cost = column(data, "unit_cost")
    discount = column(data, "discount_rate")
    stock = column(data, "stock_before_units")
    payment = column(data, "payment_code")
    rating = column(data, "customer_rating")

    unique_ids, id_counts = np.unique(transaction[np.isfinite(transaction)],
                                      return_counts=True)
    duplicate_ids = unique_ids[id_counts > 1]
    duplicate_rows = int(np.sum(id_counts[id_counts > 1]))

    mismatch = np.zeros(data.shape[0], dtype=bool)
    for product_number, expected_category in PRODUCT_CATEGORY.items():
        mismatch |= (product == product_number) & (category != expected_category)

    audit = {
        "Missing unit_price": int(np.isnan(price).sum()),
        "Missing unit_cost": int(np.isnan(cost).sum()),
        "Missing customer_rating": int(np.isnan(rating).sum()),
        "Duplicate transaction IDs": int(duplicate_ids.size),
        "Rows involved in duplicate IDs": duplicate_rows,
        "Invalid day_index": int((~np.isfinite(day) | (day < 1) | (day > 30) |
                                  (day != np.floor(day))).sum()),
        "Unknown outlet codes": int((~np.isin(outlet, [1, 2, 3])).sum()),
        "Unknown product codes": int((~np.isin(product, list(PRODUCT_NAMES))).sum()),
        "Category mapping mismatches": int(mismatch.sum()),
        "Invalid quantity": int((~np.isfinite(quantity) | (quantity <= 0) |
                                 (quantity != np.floor(quantity))).sum()),
        "Invalid unit price": int((np.isfinite(price) & (price <= 0)).sum()),
        "Invalid unit cost": int((np.isfinite(cost) & (cost < 0)).sum()),
        "Impossible discount": int((np.isfinite(discount) &
                                    ((discount < 0) | (discount > 1))).sum()),
        "Invalid customer rating": int((np.isfinite(rating) &
                                        ((rating < 1) | (rating > 5) |
                                         (rating != np.floor(rating)))).sum()),
        "Unknown payment codes": int((~np.isin(payment, [1, 2, 3, 4])).sum()),
        "Negative stock": int((np.isfinite(stock) & (stock < 0)).sum()),
        "Zero stock before transaction": int((np.isfinite(stock) & (stock == 0)).sum()),
        "High but valid price (over INR 300)": int((np.isfinite(price) & (price > 300)).sum()),
        "High but valid discount (over 30%)": int((np.isfinite(discount) &
                                                    (discount > 0.30) &
                                                    (discount <= 1)).sum())
    }

    for issue, count in audit.items():
        print(f"{issue:40s}: {count}")

    if duplicate_ids.size:
        print("Duplicate ID values:", duplicate_ids.astype(int))

    return audit, mismatch


# ------------------------------------------------------------
# 6. TASK C: CLEANING STRATEGY
# ------------------------------------------------------------

def clean_data(data, mismatch):
    """
    Keep the original raw array unchanged.
    Exclude invalid transactions from the main analytical view.
    Missing price/cost are excluded from calculations needing those fields.
    Missing ratings do not invalidate otherwise usable sales records.
    """
    print_section("TASK C - CLEANING STRATEGY")

    raw = data.copy()

    transaction = column(raw, "transaction_id")
    day = column(raw, "day_index")
    outlet = column(raw, "outlet_id")
    product = column(raw, "product_code")
    category = column(raw, "category_code")
    quantity = column(raw, "quantity")
    price = column(raw, "unit_price")
    cost = column(raw, "unit_cost")
    discount = column(raw, "discount_rate")
    stock = column(raw, "stock_before_units")
    payment = column(raw, "payment_code")

    # Mark every occurrence of a duplicated transaction ID as a duplicate.
    finite_ids = np.isfinite(transaction)
    ids, counts = np.unique(transaction[finite_ids], return_counts=True)
    duplicate_ids = ids[counts > 1]
    duplicate_mask = np.isin(transaction, duplicate_ids)

    invalid_mask = (
        ~finite_ids |
        duplicate_mask |
        ~np.isfinite(day) | (day < 1) | (day > 30) |
        (day != np.floor(day)) |
        ~np.isin(outlet, [1, 2, 3]) |
        ~np.isin(product, list(PRODUCT_NAMES)) |
        mismatch |
        ~np.isfinite(quantity) | (quantity <= 0) |
        (quantity != np.floor(np.nan_to_num(quantity, nan=0))) |
        ~np.isfinite(price) | (price <= 0) |
        ~np.isfinite(cost) | (cost < 0) |
        ~np.isfinite(discount) | (discount < 0) | (discount > 1) |
        ~np.isin(payment, [1, 2, 3, 4]) |
        ~np.isfinite(stock) | (stock < 0)
    )

    # High prices and high-but-valid discounts are retained but flagged.
    suspicious_mask = (
        (np.isfinite(price) & (price > 300)) |
        (np.isfinite(discount) & (discount > 0.30) & (discount <= 1)) |
        (np.isfinite(stock) & (stock == 0))
    )

    clean = raw[~invalid_mask].copy()

    print("Raw records:", len(raw))
    print("Records excluded from primary financial analysis:", int(invalid_mask.sum()))
    print("Records retained:", len(clean))
    print("Suspicious-but-plausible records retained:", int(
        np.sum(suspicious_mask & ~invalid_mask)
    ))
    print("Missing prices retained in raw data but excluded from revenue analysis:",
          int(np.isnan(price).sum()))
    print("Missing costs retained in raw data but excluded from profit analysis:",
          int(np.isnan(cost).sum()))
    print("Missing ratings are not replaced with zero:",
          int(np.isnan(column(raw, "customer_rating")).sum()))

    # The clean transaction set used for sales and revenue requires price.
    sales_valid = (
        np.isfinite(column(clean, "unit_price")) &
        np.isfinite(column(clean, "quantity")) &
        (column(clean, "unit_price") > 0) &
        (column(clean, "quantity") > 0)
    )
    sales_data = clean[sales_valid].copy()

    # Profit calculations additionally require a valid cost.
    profit_valid = np.isfinite(column(sales_data, "unit_cost")) & (
        column(sales_data, "unit_cost") >= 0
    )
    profit_data = sales_data[profit_valid].copy()

    print("Rows usable for revenue metrics:", len(sales_data))
    print("Rows usable for gross-profit metrics:", len(profit_data))

    return raw, clean, sales_data, profit_data, invalid_mask, suspicious_mask


# ------------------------------------------------------------
# 7. TASK D: METRIC ENGINE
# ------------------------------------------------------------

def calculate_metrics(data):
    quantity = column(data, "quantity")
    price = column(data, "unit_price")
    cost = column(data, "unit_cost")
    discount_rate = column(data, "discount_rate")

    gross_sales = quantity * price
    discount_amount = gross_sales * discount_rate
    net_revenue = gross_sales - discount_amount
    cogs = quantity * cost
    gross_profit = net_revenue - cogs

    margin = np.full(len(data), np.nan)
    valid_revenue = net_revenue != 0
    margin[valid_revenue] = (
        gross_profit[valid_revenue] / net_revenue[valid_revenue]
    ) * 100

    average_selling_price = np.full(len(data), np.nan)
    positive_quantity = quantity > 0
    average_selling_price[positive_quantity] = (
        net_revenue[positive_quantity] / quantity[positive_quantity]
    )

    return {
        "gross_sales": gross_sales,
        "discount_amount": discount_amount,
        "net_revenue": net_revenue,
        "cost_of_goods_sold": cogs,
        "gross_profit": gross_profit,
        "gross_margin_percent": margin,
        "average_selling_price": average_selling_price
    }


def metric_summary(data, metrics):
    print_section("TASK D - METRIC CALCULATIONS")

    for name, values in metrics.items():
        print(f"{name:28s}: INR/percent total or average = "
              f"{np.nansum(values):,.2f} / {safe_mean(values):,.2f}")

    total_revenue = safe_sum(metrics["net_revenue"])
    total_profit = safe_sum(metrics["gross_profit"])
    overall_margin = safe_percentage(total_profit, total_revenue)

    print("\nTotal gross sales (INR):", round(safe_sum(metrics["gross_sales"]), 2))
    print("Total discount amount (INR):", round(safe_sum(metrics["discount_amount"]), 2))
    print("Total net revenue (INR):", round(total_revenue, 2))
    print("Total cost of goods sold (INR):",
          round(safe_sum(metrics["cost_of_goods_sold"]), 2))
    print("Total gross profit (INR):", round(total_profit, 2))
    print("Overall gross margin (%):", round(overall_margin, 2)
          if np.isfinite(overall_margin) else "Undefined")

    return overall_margin


# ------------------------------------------------------------
# 8. TASK E: OUTLET AND PRODUCT COMPARISON
# ------------------------------------------------------------

def group_comparison(data, metrics, group_column, names, heading):
    print_section(heading)

    groups = np.unique(column(data, group_column).astype(int))
    result = []

    for group in groups:
        mask = column(data, group_column) == group
        revenue = safe_sum(metrics["net_revenue"][mask])
        profit = safe_sum(metrics["gross_profit"][mask])
        units = safe_sum(column(data, "quantity")[mask])
        margin = safe_percentage(profit, revenue)

        label = names.get(int(group), f"Unknown {group}")
        result.append([group, label, units, revenue, profit, margin])
        print(
            f"{label:24s} units={units:8.0f} "
            f"net revenue=INR {revenue:12,.2f} "
            f"gross profit=INR {profit:12,.2f} "
            f"margin={margin:7.2f}%"
        )

    result.sort(key=lambda item: item[3], reverse=True)
    print("\nRevenue ranking:")
    for rank, item in enumerate(result, start=1):
        print(f"{rank}. {item[1]} - INR {item[3]:,.2f}")

    return result


# ------------------------------------------------------------
# 9. TASK F: DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

def descriptive_statistics(data, metrics):
    print_section("TASK F - DESCRIPTIVE STATISTICS")

    selected = {
        "Quantity": column(data, "quantity"),
        "Net revenue": metrics["net_revenue"],
        "Gross profit": metrics["gross_profit"],
        "Discount rate": column(data, "discount_rate"),
        "Customer rating": column(data, "customer_rating")
    }

    for name, values in selected.items():
        valid = values[np.isfinite(values)]
        if valid.size == 0:
            print(name, ": no valid values")
            continue

        print(f"\n{name}")
        print("  Count:", valid.size)
        print("  Mean:", round(float(np.mean(valid)), 3))
        print("  Median:", round(float(np.median(valid)), 3))
        print("  Standard deviation (population):", round(float(np.std(valid)), 3))
        print("  Minimum:", round(float(np.min(valid)), 3))
        print("  Maximum:", round(float(np.max(valid)), 3))
        print("  25th percentile:", round(float(np.percentile(valid, 25)), 3))
        print("  75th percentile:", round(float(np.percentile(valid, 75)), 3))
        print("  IQR:", round(float(np.percentile(valid, 75) -
                                  np.percentile(valid, 25)), 3))

    print("\nInterpretation note: compare mean with median to identify skew.")
    print("Extreme values are investigated; they are not automatically deleted.")


# ------------------------------------------------------------
# 10. TASK G: DAILY TREND AND ANOMALY FLAGS
# ------------------------------------------------------------

def daily_trend(data, metrics):
    print_section("TASK G - DAILY REVENUE TREND")

    days = np.arange(1, 31)
    daily_revenue = np.zeros(30)
    daily_profit = np.zeros(30)
    daily_units = np.zeros(30)

    day_values = column(data, "day_index").astype(int)
    for day_number in days:
        mask = day_values == day_number
        daily_revenue[day_number - 1] = safe_sum(
            metrics["net_revenue"][mask]
        )
        daily_profit[day_number - 1] = safe_sum(
            metrics["gross_profit"][mask]
        )
        daily_units[day_number - 1] = safe_sum(
            column(data, "quantity")[mask]
        )

    revenue_change = np.full(30, np.nan)
    revenue_change[1:] = np.diff(daily_revenue)

    percentage_change = np.full(30, np.nan)
    previous = daily_revenue[:-1]
    valid_previous = previous != 0
    percentage_change[1:][valid_previous] = (
        revenue_change[1:][valid_previous] / previous[valid_previous]
    ) * 100

    moving_average = np.full(30, np.nan)
    for index in range(2, 30):
        moving_average[index] = np.mean(daily_revenue[index - 2:index + 1])

    # Flag days with an absolute revenue change beyond 2 standard deviations.
    valid_changes = revenue_change[np.isfinite(revenue_change)]
    change_mean = np.mean(valid_changes)
    change_std = np.std(valid_changes)
    unusual = np.zeros(30, dtype=bool)
    unusual[1:] = np.abs(revenue_change[1:] - change_mean) > (2 * change_std)

    print("Day | Revenue | Change | Change % | 3-day average | Flag")
    for index, day_number in enumerate(days):
        change_text = "-" if not np.isfinite(revenue_change[index]) else f"{revenue_change[index]:.2f}"
        pct_text = "-" if not np.isfinite(percentage_change[index]) else f"{percentage_change[index]:.2f}%"
        avg_text = "-" if not np.isfinite(moving_average[index]) else f"{moving_average[index]:.2f}"
        flag = "REVIEW" if unusual[index] else ""
        print(f"{day_number:3d} | {daily_revenue[index]:10.2f} | "
              f"{change_text:>10s} | {pct_text:>9s} | {avg_text:>13s} | {flag}")

    rows = []
    for index, day_number in enumerate(days):
        rows.append([
            day_number, daily_revenue[index], daily_profit[index],
            daily_units[index], revenue_change[index],
            percentage_change[index], moving_average[index],
            int(unusual[index])
        ])

    save_numeric_report(
        "freshflow_daily_trend.csv",
        ["day_index", "net_revenue", "gross_profit", "units_sold",
         "revenue_change", "revenue_percentage_change",
         "three_day_moving_average", "unusual_day_flag"],
        rows
    )
    return daily_revenue, unusual


# ------------------------------------------------------------
# 11. TASK H: DISCOUNT INVESTIGATION
# ------------------------------------------------------------

def discount_analysis(data, metrics):
    print_section("TASK H - DISCOUNT INVESTIGATION")

    discount = column(data, "discount_rate")
    bands = [
        ("0% (no discount)", discount == 0),
        ("Above 0% to 10%", (discount > 0) & (discount <= 0.10)),
        ("Above 10% to 20%", (discount > 0.10) & (discount <= 0.20)),
        ("Above 20% to 30%", (discount > 0.20) & (discount <= 0.30)),
        ("Above 30% (valid high)", (discount > 0.30) & (discount <= 1))
    ]

    print("Discount band | Transactions | Units | Net revenue | Gross profit | Margin")
    for label, mask in bands:
        count = int(mask.sum())
        units = safe_sum(column(data, "quantity")[mask])
        revenue = safe_sum(metrics["net_revenue"][mask])
        profit = safe_sum(metrics["gross_profit"][mask])
        margin = safe_percentage(profit, revenue)
        print(f"{label:23s} | {count:12d} | {units:5.0f} | "
              f"{revenue:11.2f} | {profit:12.2f} | {margin:7.2f}%")

    print("\nImportant: this is an observational comparison.")
    print("Discount groups may differ by product, outlet, time and customer mix.")
    print("The results do not prove that discounts caused changes in quantity or profit.")


# ------------------------------------------------------------
# 12. TASK I: ANOMALY REGISTER
# ------------------------------------------------------------

def anomaly_register(raw, mismatch):
    print_section("TASK I - ANOMALY REGISTER")

    quantity = column(raw, "quantity")
    price = column(raw, "unit_price")
    cost = column(raw, "unit_cost")
    discount = column(raw, "discount_rate")
    rating = column(raw, "customer_rating")
    stock = column(raw, "stock_before_units")
    outlet = column(raw, "outlet_id")
    payment = column(raw, "payment_code")
    product = column(raw, "product_code")
    transaction = column(raw, "transaction_id")

    issue_masks = {
        "MISSING_UNIT_PRICE": np.isnan(price),
        "MISSING_UNIT_COST": np.isnan(cost),
        "MISSING_CUSTOMER_RATING": np.isnan(rating),
        "INVALID_QUANTITY": np.isfinite(quantity) & (quantity <= 0),
        "NEGATIVE_UNIT_PRICE": np.isfinite(price) & (price <= 0),
        "NEGATIVE_UNIT_COST": np.isfinite(cost) & (cost < 0),
        "IMPOSSIBLE_DISCOUNT": np.isfinite(discount) &
                              ((discount < 0) | (discount > 1)),
        "INVALID_RATING": np.isfinite(rating) &
                          ((rating < 1) | (rating > 5)),
        "PRODUCT_CATEGORY_MISMATCH": mismatch,
        "UNKNOWN_OUTLET": ~np.isin(outlet, [1, 2, 3]),
        "UNKNOWN_PAYMENT": ~np.isin(payment, [1, 2, 3, 4]),
        "HIGH_PRICE_REVIEW": np.isfinite(price) & (price > 300),
        "HIGH_DISCOUNT_REVIEW": np.isfinite(discount) &
                                (discount > 0.30) & (discount <= 1),
        "ZERO_STOCK_REVIEW": np.isfinite(stock) & (stock == 0)
    }

    rows = []
    for issue, mask in issue_masks.items():
        for index in np.where(mask)[0]:
            rows.append([
                int(transaction[index]) if np.isfinite(transaction[index]) else "",
                issue,
                int(column(raw, "day_index")[index])
                if np.isfinite(column(raw, "day_index")[index]) else "",
                int(outlet[index]) if np.isfinite(outlet[index]) else "",
                int(product[index]) if np.isfinite(product[index]) else ""
            ])

    print("Issue type counts:")
    for issue, mask in issue_masks.items():
        print(f"{issue:30s}: {int(mask.sum())}")

    save_numeric_report(
        "freshflow_anomaly_register.csv",
        ["transaction_id", "reason_code", "day_index", "outlet_id", "product_code"],
        rows
    )
    print("Note: one transaction can have more than one reason code.")


# ------------------------------------------------------------
# 13. TASK J: ALTERNATIVE DISCOUNT SCENARIOS
# ------------------------------------------------------------

def scenario_analysis(data, metrics):
    print_section("TASK J - DISCOUNT SCENARIO ANALYSIS")

    current_discount = column(data, "discount_rate")
    quantity = column(data, "quantity")
    price = column(data, "unit_price")
    cost = column(data, "unit_cost")

    gross_sales = quantity * price
    baseline_revenue = safe_sum(metrics["net_revenue"])
    baseline_profit = safe_sum(metrics["gross_profit"])

    # Compare alternative discount rates while keeping quantity and cost fixed.
    scenarios = [
        ("Current recorded discounts", current_discount),
        ("Flat 10% discount", np.full(len(data), 0.10)),
        ("Flat 15% discount", np.full(len(data), 0.15)),
        ("Flat 20% discount", np.full(len(data), 0.20))
    ]

    print("Scenario | Net revenue | Gross profit | Revenue change | Profit change")
    for label, scenario_discount in scenarios:
        scenario_revenue_array = gross_sales * (1 - scenario_discount)
        scenario_profit_array = scenario_revenue_array - (quantity * cost)
        revenue = safe_sum(scenario_revenue_array)
        profit = safe_sum(scenario_profit_array)
        revenue_change = revenue - baseline_revenue
        profit_change = profit - baseline_profit
        print(f"{label:27s} | {revenue:11,.2f} | {profit:12,.2f} | "
              f"{revenue_change:13,.2f} | {profit_change:12,.2f}")

    print("\nAssumption: quantity, product mix and unit cost remain unchanged.")
    print("These are static scenarios, not forecasts or demand predictions.")


# ------------------------------------------------------------
# 14. TASK K: VALIDATION TESTS
# ------------------------------------------------------------

def validation_tests(data, metrics, raw_shape):
    print_section("TASK K - VALIDATION TESTS")

    # Check the original CSV dimensions separately from the cleaned subset.
    assert raw_shape == (5000, 12), "Source CSV must contain 5,000 rows and 12 columns."
    assert data.ndim == 2, "Analytical data must be a two-dimensional array."
    assert data.shape[1] == 12, "Analytical data must retain all 12 columns."
    print("Source CSV shape validated:", raw_shape)
    print("Profit-valid analytical shape:", data.shape)

    # Verify metric formula using at least five sample records.
    print("Five-row manual metric check:")
    for index in range(5):
        q = column(data, "quantity")[index]
        p = column(data, "unit_price")[index]
        c = column(data, "unit_cost")[index]
        d = column(data, "discount_rate")[index]

        gross = q * p
        discount_value = gross * d
        net = gross - discount_value
        cogs = q * c
        profit = net - cogs

        print(f"Row {index + 1}: gross={gross:.2f}, discount={discount_value:.2f}, "
              f"net={net:.2f}, COGS={cogs:.2f}, profit={profit:.2f}")

        np.testing.assert_allclose(metrics["gross_sales"][index], gross)
        np.testing.assert_allclose(metrics["discount_amount"][index], discount_value)
        np.testing.assert_allclose(metrics["net_revenue"][index], net)
        np.testing.assert_allclose(metrics["cost_of_goods_sold"][index], cogs)
        np.testing.assert_allclose(metrics["gross_profit"][index], profit)

    # NaN and zero-denominator safety.
    assert np.isnan(safe_percentage(10, 0))
    assert np.isnan(safe_mean(np.array([np.nan, np.nan])))

    # Check for finite values in calculated arrays where inputs are valid.
    assert np.all(np.isfinite(metrics["gross_sales"]))
    assert np.all(np.isfinite(metrics["net_revenue"]))
    assert np.all(np.isfinite(metrics["gross_profit"]))

    # Check invalid shape detection.
    try:
        assert data.shape[1] == 11
    except AssertionError:
        print("Invalid-shape test: correctly detected that 12 columns are present.")

    print("\nAll metric formula and numerical safety tests passed.")


# ------------------------------------------------------------
# 15. TASK L: EXECUTIVE RECOMMENDATIONS
# ------------------------------------------------------------

def executive_recommendations(outlet_results, product_results, daily_revenue,
                               overall_margin):
    print_section("TASK L - EXECUTIVE SUMMARY AND RECOMMENDATIONS")

    best_outlet = max(outlet_results, key=lambda item: item[3]) if outlet_results else None
    best_product = max(product_results, key=lambda item: item[3]) if product_results else None
    highest_units_product = max(product_results, key=lambda item: item[2]) if product_results else None
    lowest_day = int(np.argmin(daily_revenue)) + 1
    highest_day = int(np.argmax(daily_revenue)) + 1

    print("Evidence-based findings:")
    if best_outlet:
        print(f"1. Highest-revenue outlet: {best_outlet[1]} "
              f"(INR {best_outlet[3]:,.2f}).")
    if best_product:
        print(f"2. Highest-revenue product: {best_product[1]} "
              f"(INR {best_product[3]:,.2f}).")
    if highest_units_product:
        print(f"3. Highest-volume product: {highest_units_product[1]} "
              f"({highest_units_product[2]:.0f} units).")
    print(f"4. Highest daily revenue: Day {highest_day}; "
          f"lowest daily revenue: Day {lowest_day}.")
    print(f"5. Overall gross margin: {overall_margin:.2f}%.")

    print("\nRecommended actions:")
    print("- Review outlet-level product mix and share practices from the strongest outlet.")
    print("- Investigate high-discount transactions and monitor gross margin by discount band.")
    print("- Check high-price, zero-stock and data-quality exceptions before operational action.")
    print("- Track daily net revenue and the three-day moving average.")
    print("- Recheck the same metrics after another observation period.")

    print("\nLimitations:")
    print("- The dataset covers only 30 consecutive days.")
    print("- It cannot establish annual seasonality or long-term business performance.")
    print("- Gross profit excludes rent, salaries, utilities and other operating expenses.")
    print("- Correlation and group differences do not prove causation.")
    print("- Transaction records alone cannot establish inventory waste or stock flow.")


# ------------------------------------------------------------
# 16. MAIN PROGRAM: RUN THE COMPLETE ASSIGNMENT
# ------------------------------------------------------------

def main():
    print_section("FRESHFLOW JUICE COMPANY - NUMPY DATA ANALYST ASSIGNMENT")

    # Step 1: Generate the source CSV file.
    generate_freshflow_csv(CSV_FILE)

    # Step 2: Load the CSV using NumPy.
    raw_data = load_data(CSV_FILE)

    # Step 3: Inspect and audit the original data.
    data_profile(raw_data)
    audit, category_mismatch = integrity_audit(raw_data)

    # Step 4: Create cleaned analytical arrays.
    raw, clean, sales_data, profit_data, invalid_mask, suspicious_mask = (
        clean_data(raw_data, category_mismatch)
    )

    # Step 5: Calculate metrics for the common valid financial record set.
    # Revenue metrics use valid price/quantity rows.
    sales_metrics = calculate_metrics(sales_data)

    # Profit metrics are calculated on records with valid cost as well.
    profit_metrics = calculate_metrics(profit_data)

    metric_summary(profit_data, profit_metrics)

    # Step 6: Compare outlets and products on the same profit-valid record set.
    outlet_results = group_comparison(
        profit_data, profit_metrics, "outlet_id", OUTLET_NAMES,
        "TASK E - OUTLET PERFORMANCE"
    )
    product_results = group_comparison(
        profit_data, profit_metrics, "product_code", PRODUCT_NAMES,
        "TASK E - PRODUCT PERFORMANCE"
    )

    # Category comparison is also useful for management.
    group_comparison(
        profit_data, profit_metrics, "category_code", CATEGORY_NAMES,
        "ADDITIONAL - CATEGORY PERFORMANCE"
    )

    # Step 7: Descriptive statistics and chronological trend.
    descriptive_statistics(profit_data, profit_metrics)
    daily_revenue, unusual_days = daily_trend(sales_data, sales_metrics)

    # Step 8: Discount investigation, anomaly register and scenarios.
    discount_analysis(profit_data, profit_metrics)
    anomaly_register(raw_data, category_mismatch)
    scenario_analysis(profit_data, profit_metrics)

    # Step 9: Numerical validation and management recommendations.
    validation_tests(profit_data, profit_metrics, raw_data.shape)
    overall_margin = safe_percentage(
        safe_sum(profit_metrics["gross_profit"]),
        safe_sum(profit_metrics["net_revenue"])
    )
    executive_recommendations(
        outlet_results, product_results, daily_revenue, overall_margin
    )

    print_section("ASSIGNMENT RUN COMPLETED")
    print("Main dataset:", CSV_FILE)
    print("Additional outputs:")
    print("- freshflow_daily_trend.csv")
    print("- freshflow_anomaly_register.csv")
    print("Raw data was preserved; cleaned arrays were created separately.")
    print("No Pandas library was used.")


if __name__ == "__main__":
    main()
