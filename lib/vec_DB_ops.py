from pymilvus import Collection, FieldSchema, DataType, connections


if __name__ == '__main__':
    connections.connect(alias='default', host='localhost', port='3000')
