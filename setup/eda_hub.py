"""
setup/eda_hub.py

FoodHub EDA + Feature Engineering Pipeline

Input:
    data/foodhub_order.csv

Outputs:
    processed/*.csv
    processed/foodhub_order_output.csv
    processed/foodhub.db

Author:
    FoodHub Agent Platform
"""

from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd


class FoodHubEDA:

    def __init__(
        self,
        input_file="data/foodhub_order.csv",
        output_dir="processed",
        db_name="foodhub.db",
    ):

        self.input_file = Path(input_file)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.output_dir / db_name

        self.df = None

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    def save_step(self, df, step_name):
        filepath = self.output_dir / f"foodhub_order_{step_name}.csv"
        df.to_csv(filepath, index=False)

        print(f"[SAVED] {filepath}")

    # ---------------------------------------------------------
    # Step 1 : Load
    # ---------------------------------------------------------

    def load_data(self):

        print("\n[STEP] Load Data")

        self.df = pd.read_csv(self.input_file)

        self.save_step(
            self.df,
            "loaded"
        )

        return self.df

    # ---------------------------------------------------------
    # Step 2 : Basic Cleaning
    # ---------------------------------------------------------

    def clean_data(self):

        print("\n[STEP] Data Cleaning")

        df = self.df.copy()

        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        df["rating"] = (
            df["rating"]
            .replace("Not given", np.nan)
        )

        df["rating"] = pd.to_numeric(
            df["rating"],
            errors="coerce"
        )

        df.drop_duplicates(
            subset=["order_id"],
            inplace=True
        )

        self.df = df

        self.save_step(
            df,
            "cleaned"
        )

        return df

    # ---------------------------------------------------------
    # Step 3 : Type Standardization
    # ---------------------------------------------------------

    def standardize_types(self):

        print("\n[STEP] Standardize Types")

        df = self.df.copy()

        numeric_cols = [
            "cost_of_the_order",
            "food_preparation_time",
            "delivery_time",
            "rating"
        ]

        for col in numeric_cols:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        self.df = df

        self.save_step(
            df,
            "datatypes"
        )

        return df

    # ---------------------------------------------------------
    # Step 4 : Missing Handling
    # ---------------------------------------------------------

    def handle_missing(self):

        print("\n[STEP] Handle Missing Values")

        df = self.df.copy()

        df["rating_missing"] = (
            df["rating"].isna().astype(int)
        )

        self.df = df

        self.save_step(
            df,
            "missing_handled"
        )

        return df

    # ---------------------------------------------------------
    # Step 5 : Feature Engineering
    # ---------------------------------------------------------

    def feature_engineering(self):

        print("\n[STEP] Feature Engineering")

        df = self.df.copy()

        # ---------------------------------
        # Total time
        # ---------------------------------

        df["total_time"] = (
            df["food_preparation_time"]
            + df["delivery_time"]
        )

        # ---------------------------------
        # Revenue
        # ---------------------------------

        df["commission"] = np.where(
            df["cost_of_the_order"] > 20,
            df["cost_of_the_order"] * 0.25,
            np.where(
                df["cost_of_the_order"] > 5,
                df["cost_of_the_order"] * 0.15,
                0
            )
        )

        # ---------------------------------
        # Price Band
        # ---------------------------------

        df["price_band"] = pd.cut(
            df["cost_of_the_order"],
            bins=[
                0,
                10,
                15,
                20,
                100
            ],
            labels=[
                "LOW",
                "MID_LOW",
                "MID",
                "PREMIUM"
            ],
        )

        # ---------------------------------
        # SLA Labels
        # ---------------------------------

        df["delivery_sla"] = np.where(
            df["delivery_time"] <= 23,
            "FAST",
            np.where(
                df["delivery_time"] <= 30,
                "STANDARD",
                "SLOW"
            )
        )

        # ---------------------------------
        # Total Time SLA
        # ---------------------------------

        df["total_time_sla"] = np.where(
            df["total_time"] > 60,
            "BREACHED",
            "WITHIN_SLA"
        )

        self.df = df

        self.save_step(
            df,
            "feature_engineered"
        )

        return df

    # ---------------------------------------------------------
    # Step 6 : Customer Features
    # ---------------------------------------------------------

    def customer_features(self):

        print("\n[STEP] Customer Features")

        df = self.df.copy()

        customer_orders = (
            df.groupby("customer_id")
            .size()
            .reset_index(name="order_count")
        )

        def segment(order_count):

            if order_count > 10:
                return "FREQUENT"

            elif order_count >= 5:
                return "REGULAR"

            elif order_count >= 2:
                return "NEW"

            return "FIRST_TIME"

        customer_orders["customer_segment"] = (
            customer_orders["order_count"]
            .apply(segment)
        )

        df = df.merge(
            customer_orders,
            on="customer_id",
            how="left"
        )

        self.df = df

        self.save_step(
            df,
            "customer_segmented"
        )

        return df

    # ---------------------------------------------------------
    # Step 7 : Cuisine Features
    # ---------------------------------------------------------

    def cuisine_features(self):

        print("\n[STEP] Cuisine Features")

        df = self.df.copy()

        cuisine_volume = (
            df["cuisine_type"]
            .value_counts()
        )

        def cuisine_tier(x):

            cnt = cuisine_volume[x]

            if cnt >= 100:
                return "HIGH"

            elif cnt >= 20:
                return "MEDIUM"

            return "LOW"

        df["cuisine_tier"] = (
            df["cuisine_type"]
            .apply(cuisine_tier)
        )

        self.df = df

        self.save_step(
            df,
            "cuisine_enriched"
        )

        return df

    # ---------------------------------------------------------
    # Step 8 : Restaurant Features
    # ---------------------------------------------------------

    def restaurant_features(self):

        print("\n[STEP] Restaurant Features")

        df = self.df.copy()

        restaurant_orders = (
            df["restaurant_name"]
            .value_counts()
        )

        df["restaurant_order_count"] = (
            df["restaurant_name"]
            .map(restaurant_orders)
        )

        self.df = df

        self.save_step(
            df,
            "restaurant_enriched"
        )

        return df

    # ---------------------------------------------------------
    # Step 9 : Rating Imputation
    # ---------------------------------------------------------

    def impute_rating(self):

        print("\n[STEP] Rating Imputation")

        df = self.df.copy()

        mean_rating = (
            df["rating"]
            .dropna()
            .mean()
        )

        df["rating"] = (
            df["rating"]
            .fillna(round(mean_rating, 1))
        )

        self.df = df

        self.save_step(
            df,
            "rating_imputed"
        )

        return df

    # ---------------------------------------------------------
    # Step 10 : Final Output
    # ---------------------------------------------------------

    def finalize(self):

        print("\n[STEP] Final Output")

        final_path = (
            self.output_dir
            / "foodhub_order_output.csv"
        )

        self.df.to_csv(
            final_path,
            index=False
        )

        print(f"[FINAL CSV] {final_path}")

        return self.df

    # ---------------------------------------------------------
    # Step 11 : SQLite
    # ---------------------------------------------------------

    def save_sqlite(self):

        print("\n[STEP] Save SQLite")

        conn = sqlite3.connect(
            self.db_path
        )

        self.df.to_sql(
            "foodhub_order_output",
            conn,
            index=False,
            if_exists="replace"
        )

        conn.close()

        print(
            f"[SQLITE] table overwritten : foodhub_order_output"
        )

    # ---------------------------------------------------------
    # Pipeline
    # ---------------------------------------------------------

    def run(self):

        self.load_data()
        self.clean_data()
        self.standardize_types()
        self.handle_missing()
        self.feature_engineering()
        self.customer_features()
        self.cuisine_features()
        self.restaurant_features()
        self.impute_rating()
        self.finalize()
        self.save_sqlite()

        return self.df