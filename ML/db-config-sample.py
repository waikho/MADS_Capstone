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