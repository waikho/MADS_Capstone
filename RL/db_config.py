pgSecrets = {
    'user' : 'capstone',
    'password' : '8SbrTAlYOu',
    'host' : '43.138.250.245',
    'dbname' : 'capstone'
}

def pgDictToConn(secretDict):
    pgStrs = []
    for key in secretDict:
        pgStrs.append('{}={}'.format(key, secretDict[key]))
    return ' '.join(pgStrs)

pgConnStr = pgDictToConn(pgSecrets)