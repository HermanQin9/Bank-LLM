import psycopg2
from psycopg2.extras import RealDictCursor

conn = psycopg2.connect(
    dbname="frauddb",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5432
)

cursor = conn.cursor(cursor_factory=RealDictCursor)

# Get customers table structure
cursor.execute("""
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_name = 'customers'
    ORDER BY ordinal_position
""")

print("\n=== CUSTOMERS TABLE STRUCTURE ===")
for row in cursor.fetchall():
    nullable = "NULL" if row['is_nullable'] == 'YES' else "NOT NULL"
    print(f"  {row['column_name']:30} {row['data_type']:20} {nullable}")

cursor.close()
conn.close()
