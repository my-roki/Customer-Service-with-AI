import mysql.connector
from configparser import ConfigParser
from mysql.connector import MySQLConnection, Error
from DB_Base import *

def user_history(user_no):
    try:
        dbconfig = read_db_config()
        conn = MySQLConnection(**dbconfig)
        cursor = conn.cursor(prepared=True)
        history_sql ="""select date_format(reg_time, '%Y-%m-%d'), product_name, order_amount, user_name
                        from order_table as ot left join user_table as ut on ot.user_no = ut.user_no
                        left join menu_table as mt on ot.product_no = mt.product_no
                        where ot.user_no = %s
                        order by reg_time DESC limit 6;
                        """
        history_id = (user_no,)
        cursor.execute(history_sql, history_id)
        rows = cursor.fetchall()
        if rows == []:
            rows = [["-","-","-","신규고객님"]]
        print('Total Row(s):', cursor.rowcount)
        return rows
    except Error as e:
        print(e)

    finally:
        cursor.close()
        conn.close()