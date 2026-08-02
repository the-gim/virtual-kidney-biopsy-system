import duckdb
import os

DATA_DIR = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data"
CHARTEVENTS_GZ = os.path.join(DATA_DIR, "chartevents.csv.gz")
LABEVENTS_CSV = os.path.join(DATA_DIR, "labevents.csv")
LABELS_CSV = os.path.join(DATA_DIR, "biopsy_labels.csv")

OUT_CHARTEVENTS_PQ = os.path.join(DATA_DIR, "filtered_chartevents.parquet")
OUT_LABEVENTS_PQ = os.path.join(DATA_DIR, "filtered_labevents.parquet")

def main():
    print("Connecting to DuckDB...")
    con = duckdb.connect(database=':memory:')

    # Item IDs relevant to kidney function and vitals
    # Lab Events: Creatinine(50912), BUN(51006), Albumin(50862), Bicarbonate(50882), Sodium(50983), Potassium(50971)
    kidney_lab_items = "50912, 51006, 50862, 50882, 50983, 50971, 51081, 51100"
    
    # Chart Events: HR(220045), NBP Sys(220179), NBP Dias(220180), NBP Mean(220181), Creatinine(220615), BUN(225624)
    kidney_chart_items = "220045, 220179, 220180, 220181, 220615, 225624, 227442, 227443"

    print("Checking if labels file exists...")
    if not os.path.exists(LABELS_CSV):
        print(f"Error: {LABELS_CSV} does not exist. Please run 01_extract_labels_nlp.py first.")
        return

    # 1. Filter Lab Events
    print(f"Filtering {LABEVENTS_CSV}...")
    query_lab = f"""
    COPY (
        SELECT l.*
        FROM read_csv_auto('{LABEVENTS_CSV}') AS l
        JOIN read_csv_auto('{LABELS_CSV}') AS b ON l.subject_id = b.subject_id
        WHERE l.itemid IN ({kidney_lab_items})
    ) TO '{OUT_LABEVENTS_PQ}' (FORMAT PARQUET);
    """
    con.execute(query_lab)
    print(f"Saved filtered lab events to {OUT_LABEVENTS_PQ}")

    # 2. Filter Chart Events (compressed gz)
    print(f"Filtering {CHARTEVENTS_GZ}...")
    query_chart = f"""
    COPY (
        SELECT c.*
        FROM read_csv_auto('{CHARTEVENTS_GZ}') AS c
        JOIN read_csv_auto('{LABELS_CSV}') AS b ON c.subject_id = b.subject_id
        WHERE c.itemid IN ({kidney_chart_items})
    ) TO '{OUT_CHARTEVENTS_PQ}' (FORMAT PARQUET);
    """
    con.execute(query_chart)
    print(f"Saved filtered chart events to {OUT_CHARTEVENTS_PQ}")
    
    print("Filtering complete!")

if __name__ == "__main__":
    main()
