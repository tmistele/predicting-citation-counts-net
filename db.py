import MySQLdb
import settings

_connection = None


def db():
    global _connection
    if not _connection:
        print("Connecting to DB")
        _connection = MySQLdb.connect(
            host=settings.DB_HOST,
            user=settings.DB_USER,
            passwd=settings.DB_PASS,
            db=settings.DB_NAME
        )
    return _connection
