import psycopg2

conn = psycopg2.connect(
    host='localhost',
    port=5432,
    database='frauddb',
    user='postgres',
    password='postgres'
)

cur = conn.cursor()

# Check transactions table
cur.execute("""
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name = 'transactions' 
    ORDER BY ordinal_position
""")
cols = [r[0] for r in cur.fetchall()]
print('Transactions table columns:')
for col in cols:
    print(f'  - {col}')

# Check customer_profiles table
cur.execute("""
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name = 'customer_profiles' 
    ORDER BY ordinal_position
""")
cols = [r[0] for r in cur.fetchall()]
print('\nCustomer_profiles table columns:')
for col in cols:
    print(f'  - {col}')

# Check fraud_alerts table
cur.execute("""
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name = 'fraud_alerts' 
    ORDER BY ordinal_position
""")
cols = [r[0] for r in cur.fetchall()]
print('\nFraud_alerts table columns:')
for col in cols:
    print(f'  - {col}')

conn.close()
