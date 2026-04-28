# generate_pins_mysql_unique.py
import mysql.connector
import secrets
import os

db_conf = {
    'host': 'localhost',
    'user': 'root',
    'password': '',   # ضع الباسورد لو عندك
    'database': 'vending_app'
}

def gen_pin():
    return '{:06d}'.format(secrets.randbelow(10**6))

def main(n=100):
    conn = mysql.connector.connect(**db_conf)
    cur = conn.cursor()
    inserted = 0
    for _ in range(n):
        while True:
            pin = gen_pin()
            cur.execute("SELECT COUNT(*) FROM pins WHERE plain_pin=%s", (pin,))
            if cur.fetchone()[0] == 0:
                cur.execute("INSERT INTO pins (plain_pin) VALUES (%s)", (pin,))
                print("Inserted PIN:", pin)
                inserted += 1
                break
    conn.commit()
    cur.close()
    conn.close()
    print("Done: inserted", inserted, "unique pins")

if __name__ == '__main__':
    main(100)
