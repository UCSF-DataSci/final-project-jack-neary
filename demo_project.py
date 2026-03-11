import duckdb

con = duckdb.connect()
con.execute("INSTALL httpfs;")
con.execute("LOAD httpfs;")

result = con.execute("""
    SELECT *
    FROM read_csv_auto('https://physionet.org/files/mimic-iv-demo/2.2/hosp/patients.csv.gz')
    LIMIT 5
""").df()

print(result)