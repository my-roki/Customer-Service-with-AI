import mysql.connector
from configparser import ConfigParser
from mysql.connector import MySQLConnection, Error
from DB_Base import *


def timely_customer_count(): # 시간별 매출 건수 호출 함수
    try:
        dbconfig = read_db_config()
        conn = MySQLConnection(**dbconfig)
        cursor = conn.cursor()
        cursor.execute("""select date_format(reg_time, '%H') as time,
                                 count(order_amount) as customer
                          from order_table
                          where date_format(reg_time,'%Y-%m-%d') = CURDATE()
                          group by time;
                        """)
        rows = cursor.fetchall()
        print('Total Row(s):', cursor.rowcount)
            
        return rows
    except Error as e:
        print(e)

    finally:
        cursor.close()
        conn.close()


def today_count():
    try:
        dbconfig = read_db_config()
        conn = MySQLConnection(**dbconfig)
        cursor = conn.cursor()
        cursor.execute("""select count(*), sum(order_amount) as today from order_table where date_format(reg_time,'%Y-%m-%d') = CURDATE();
                        """)
        row = cursor.fetchone()
        print('Total Row(s):', cursor.rowcount)
        return row
    except Error as e:
        print(e)

    finally:
        cursor.close()
        conn.close()


def week_count():
    try:
        dbconfig = read_db_config()
        conn = MySQLConnection(**dbconfig)
        cursor = conn.cursor()
        cursor.execute("""select count(*), sum(order_amount) as thisweek from order_table where date_format(reg_time, '%U') = date_format(CURDATE(),'%U');
                        """)
        row = cursor.fetchone()
        print('Total Row(s):', cursor.rowcount)
        return row
    except Error as e:
        print(e)

    finally:
        cursor.close()
        conn.close()

def month_count():
    try:
        dbconfig = read_db_config()
        conn = MySQLConnection(**dbconfig)
        cursor = conn.cursor()
        cursor.execute("""select count(*), sum(order_amount) as thismonth from order_table where date_format(reg_time, '%M') = date_format(CURDATE(),'%M');
                        """)
        row = cursor.fetchone()
        print('Total Row(s):', cursor.rowcount)
        return row
    except Error as e:
        print(e)

    finally:
        cursor.close()
        conn.close()


def menu_count():
    try:
        dbconfig = read_db_config()
        conn = MySQLConnection(**dbconfig)
        cursor = conn.cursor()
        cursor.execute("""select product_name, product_price, count(*)
                          from Order_Table as ot inner join
                               Menu_Table as mt on ot.product_no = mt.product_no
                          where date_format(reg_time,'%Y-%m-%d') = CURDATE()
                          group by product_name;
                          """)
        rows = cursor.fetchall()
        print('Total Row(s):', cursor.rowcount)
        return rows
    except Error as e:
        print(e)

    finally:
        cursor.close()
        conn.close()


def login(id, pwd):
    try:
        dbconfig = read_db_config()
        conn = MySQLConnection(**dbconfig)
        cursor = conn.cursor(prepared=True)
        login_sql = """select user_id, user_pwd
                        from user_table
                        where user_id = %s;
                        """
        login_id = (id,)
        cursor.execute(login_sql, login_id)
        row = cursor.fetchone()
        if row[1] == pwd:
            result = True
        else:
            result = False
    except Error as e:
        print(e)

    finally:
        cursor.close()
        conn.close()
        return result

if __name__ == '__main__':
    login("roki@gmail.com","1111")
