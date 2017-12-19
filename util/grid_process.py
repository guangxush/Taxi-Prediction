# coding:utf-8


# 统计寻客点网格坐标并将坐标转换成id
def grid_process(input_file, output_file):
    fr = open(input_file, 'r')
    fw = open(output_file, 'w')
    records = []
    grid_hot = {}
    grid_items = {}
    i = 1
    for line in fr:
        records.append(line.strip().split(','))
    for record in records:
        grid_item = record[1]  # 网格坐标
        if grid_item in grid_hot:  # 统计网格热度
            grid_hot[grid_item] += 1
        else:
            grid_hot[grid_item] = 1
        if grid_item not in grid_items:  # 给网格编号
            grid_items[grid_item] = i
            i += 1

    outlines = []
    for record in records:
        grid_xy = record[1]  # 网格坐标
        grid_hot_item = grid_hot[grid_xy]
        grid_id = grid_items[grid_xy]
        outline = ','.join(record + [str(grid_hot_item), str(grid_id)])+ '\n'
        outlines.append(outline)
    fw.writelines(outlines)


if __name__ == '__main__':
    grid_process(input_file='../data/train_input.csv', output_file='../data/train.csv')