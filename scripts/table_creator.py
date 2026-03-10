import sqlite3

def minimal_create_database():
    """The bare minimum to create a new sqlite database."""

    # Connections connect to a database
    # For sqlite, if the database doesn't exist, it will be created
    connection = sqlite3.connect('data/article_grant_db.sqlite')

    # We can run a query directly, but we usually use pandas
    # We just wanted to check it worked
    print(connection.execute('SELECT sqlite_version();').fetchall())

    # print(connection.excute("""CREATE TABLE IF NOT EXISTS grants(
    #                         id INTEGER PRIMARY KEY AUTOINCREMENT,
    #                         application_id VARCHAR(20) NOT NULL,
    #                         start_at DATE,
    #                         grant_type VARCHAR(10),
    #                         total_cost INTEGER
    #                         );"""))

    # There exists cursors, which you can think of as Python generators
    # Ignore for now
    cursor = connection.cursor()
    cursor.execute('SELECT something FROM somewhere')

minimal_create_database()