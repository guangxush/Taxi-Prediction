# coding:utf8
import os
import sys
import math
import time


# 根据经纬度将出租车坐标转换成网格坐标
def grid_index(lon, lat):
    base = (121.31, 31.08)
    size = 100
    x = lon - base[0]
    y = lat - base[1]
    x *= 96000
    y *= 111000
    return float('%.2f' % math.ceil(x/size)), float('%.2f' % math.ceil(y/size))


# 根据时间处理寻客时间
def time_process(time):
    hour = float(time[0:2])
    minute = float(time[2:4])
    #  second = float(time[-2:])
    result = hour + (minute / 60.0)
    return float('%.1f' % result)


# 判断两点是否符合寻客点的定义
def judge(next_grid, last_grid):
    x1, y1 = next_grid[0], next_grid[1]
    x2, y2 = last_grid[0], last_grid[1]
    if abs(x2 - x1) <= 10 and abs(y2 - y1) <= 10:
        return True
    else:
        return False


# 给出出租车的出行方向
def duration(next_grid, last_grid):
    x1, y1 = next_grid[0], next_grid[1]
    x2, y2 = last_grid[0], last_grid[1]
    if x2 - x1 > 0:
        if y2 - y1 > 0:
            return 'c'
        elif y2 - y1 == 0:
            return 'f'
        elif y2 - y1 < 0:
            return 'i'
    elif x2 - x1 == 0:
        if y2 - y1 > 0:
            return 'b'
        elif y2 - y1 == 0:
            return 'e'
        elif y2 - y1 < 0:
            return 'h'
    elif x2 - x1 < 0:
        if y2 - y1 > 0:
            return 'a'
        elif y2 - y1 == 0:
            return 'd'
        elif y2 - y1 < 0:
            return 'g'


# 计算经过的时间，单位s
def during_time(start_time, end_time):
    return int(end_time)-int(start_time)


# 根据日期查询周几,输入：日期(2017-08-01),输出：周几(1-7)
def query_week(date_str):
    if int(time.strftime("%w", time.strptime(date_str, "%Y-%m-%d"))) == 0:
        return 7
    return int(time.strftime("%w", time.strptime(date_str, "%Y-%m-%d")))


# 坐标变化
def change_xy(next_grid, last_grid):
    x1, y1 = next_grid[0], next_grid[1]
    x2, y2 = last_grid[0], last_grid[1]
    return str(x2 - x1)+'x'+str(y2 - y1)+'y'


def generate_train(input_file, output_file):
    fr = open(input_file, 'r')
    fw = open(output_file, 'w')
    record = []
    for line in fr:
        line = line.strip().split(',')
        date = line[0]  # 寻客日期0
        time = line[1]  # 寻客时间1
        week = query_week(line[-1][:10])  # 寻客周几 2
        taxi_id = line[3]  # 出租车ID3
        lon = line[4]  # 所在经度4
        lat = line[5]  # 所在纬度5
        status = line[-3]  # 是否载客6
        temp_line = ','.join([date, time, str(week), taxi_id, lon, lat, status])
        record.append(temp_line)

    outlines = []
    i = 0
    while i < len(record)-1:
        current_line = record[i].strip().split(',')
        next_line = record[i+1].strip().split(',')
        if current_line[-1] == '1' and next_line[-1] == '0':
            start_work = current_line  # start_word是寻客开始点
            start_geo = grid_index(float(start_work[4]), float(start_work[5]))
            i += 1
            while current_line[-1] == '0' and i < len(record)-1:  # 寻找下一个载客点
                i += 1
            end_work = record[i].strip().split(',')  # current_line是寻客结束点
            end_geo = grid_index(float(end_work[4]), float(end_work[5]))
            cost_time = during_time(start_work[1], end_work[1])  # 寻客时间
            if judge(start_geo, end_geo) and cost_time > 30:
                taxi_id = end_work[3]  # 出租车ID
                parking_place = str(end_geo[0])+'-'+str(end_geo[1])  # 寻客终点坐标
                durations = duration(start_geo, end_geo)  # 出行方向的变化（九宫格）
                start_time = time_process(start_work[1])  # 开始寻客时间
                geo_change_xy = change_xy(start_geo, end_geo)  # 网格坐标的变化
                week = str(end_work[2])  # 寻客周几
                during_times = str(during_time(start_work[1], end_work[1]))+'s'  # 寻客时间
                temp_line = ','.join([taxi_id, parking_place, str(durations), str(week), str(during_times),  str(start_time),
                                      str(geo_change_xy)])+'\n'
                outlines.append(temp_line)
        else:
            i += 1
    fw.writelines(outlines)


if __name__ == '__main__':
    generate_train(input_file='../raw_data/raw_gps/10066.txt', output_file='../data/train_input.csv')