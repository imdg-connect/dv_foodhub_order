from fastapi import FastAPI
from setup.eda_hub import FoodHubEDA

app = FastAPI(
    title="FoodHub EDA API"
)

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/run-eda")
def run_eda():

    pipeline = FoodHubEDA(
        input_file="data/foodhub_order.csv",
        output_dir="processed",
        db_name="foodhub.db"
    )

    pipeline.run()

    return {
        "status": "success",
        "output_csv": "processed/foodhub_order_output.csv",
        "sqlite_db": "processed/foodhub.db"
    }
