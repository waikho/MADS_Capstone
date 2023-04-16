#Constants
EXCHANGE_OPEN_HOUR = 9
EXCHANGE_OPEN_MIN = 30
EXCHANGE_CLOSE_HOUR = 16
EXCHANGE_CLOSE_MIN = 0

pgSecrets = {
    'user' : 'DB_USER',
    'password' : 'DB _PASSWORD',
    'host' : 'DB_HOST',
    'dbname' : 'DB_NAME'
}

def pgDictToConn(secretDict):
    pgStrs = []
    for key in secretDict:
        pgStrs.append('{}={}'.format(key, secretDict[key]))
    return ' '.join(pgStrs)

pgConnStr = pgDictToConn(pgSecrets)

api = {
    'alpaca' : {
        'key' : 'YOUR_ALPACA_KEY',
        'secret' : 'YOUR_ALPACA_SECRET'
    },
    'pubproxy' : {
        'key' : 'YOUR_PUBPROXY_KEY',
        'refresh_rate' : 30
    },
    'sendgrid' : {
        'key' : 'YOUR_SENDGRID_KEY',
        'from_email' : 'YOUR_NOTIFICATION_EMAIL'
    }
}