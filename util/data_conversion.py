#coding:utf-8
import time
from generate_train import grid_index
from generate_train import time_process


def data_conversion(file_name):
    fr = open(file_name, 'r')
    fw = open('../predict/test.csv', 'a')
    raw_data = []
    for line in fr:
        raw_data.append(line.strip().split(','))
    record = raw_data[len(raw_data)-1]
    print (record)
    taxiID = record[0]
    point = grid_index(float(record[1]), float(record[2]))
    grid_point = str(point[0])+'-'+str(point[1])
    direction = record[3]  # a,b,c,d,e,f,g,h
    time_temp = time.strftime('%H%M%S', time.localtime(time.time()))
    timeData = time_process(time_temp)
    duration = record[4]
    gridID = record[5]
    outlines = ','.join([taxiID, grid_point, direction, str(timeData), duration, gridID])+'\n'
    print (outlines)
    fw.writelines(outlines)

if __name__ == '__main__':
    data_conversion(file_name='../predict/rawdata.txt')