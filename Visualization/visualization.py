import matplotlib.pyplot as plt

# 创建数据
data = [0.6892, 0.7332, 0.6771, 0.7231, 0.6733, 0.7227, 0.7040, 0.7644, 0.7075, 0.7636, 0.6988, 0.7624]  # 这里的数据表示每个条的高度

data = [0.6715, 0.7045, 0.6250, 0.6331, 0.6954, 0.6840, 0.7363, 0.8015, 0.7433, 0.7697, 0.7628, 0.7863]  # 这里的数据表示每个条的高度


fig, ax = plt.subplots(figsize=(10, 6))

width = 0.4

# 将第一和第二个条紧挨着，调整X坐标位置
bars1 = ax.bar([1, 1.4, 2.5, 2.9, 4,4.4][0 : 6 : 2], data[0 : 6 : 2], width=width,  color='skyblue', alpha = 0.3)
bars1 = ax.bar([1, 1.4, 2.5, 2.9, 4,4.4][1 : 6 : 2], data[1 : 6 : 2], width=width,  color='skyblue', alpha = 1)
bars2 = ax.bar([5.5, 5.9, 7, 7.4, 8.5, 8.9][0 : 6 : 2], data[6 : 12 : 2], width=width,  color='lightcoral', alpha=0.3)
bars2 = ax.bar([5.5, 5.9, 7, 7.4, 8.5, 8.9][1 : 6 : 2], data[7 : 12 : 2], width=width,  color='lightcoral', alpha=1)

ax.bar(3, 0, width=0.2)
ax.bar(6, 0, width=0.2)
ax.bar(9, 0, width=0.2)

# ax.set_xlabel('Groups')
ax.set_ylabel('Values')
# ax.set_title('SRCC and PLCC in Each Fold of Perception Before and After Improving')
ax.set_title('SRCC and PLCC in Each Fold of Alignment Before and After StairReward Improving')
ax.legend()

y_offsets = [0.01] * 12
x_positions = [1, 1.4, 2.5, 2.9, 4, 4.4, 5.5, 5.9, 7, 7.4, 8.5, 8.9]
for i in range(len(x_positions)):
    ax.annotate(str(data[i]), (x_positions[i], data[i]), ha='center')

for i in range(0, len(data), 2):
    improvement_rate = data[i + 1] / data[i] - 1
    ax.annotate(f'{improvement_rate:.2%}', (x_positions[i + 1], data[i + 1] + 0.03), ha='center', fontsize=10, color='green')

ax.set_xticks([1.2, 2.7, 4.2, 5.7, 7.2, 8.7])
ax.set_xticklabels(['k=1', 'k=2\nSRCC', 'k=3', 'k=1',  'k=2\nPLCC', 'k=3'])

# # 添加两条水平线，分别从最左方到中心和从中心到最右方
# left_line_y = max(data) + 0.05  # 调整水平线的高度
# right_line_y = min(data) + 0.25  # 调整水平线的高度
# plt.axhline(y=left_line_y, color='purple', alpha=0.8, linestyle='--', label='Left Line', xmin=0, xmax=4)
# plt.axhline(y=right_line_y, color='blue', alpha=0.8, linestyle='--', label='Right Line')

# 创建起点和终点坐标
x1, y1 = 0, sum(data[0 : 6 : 2])/3
x2, y2 = 4.6, sum(data[0 : 6 : 2])/3
ax.plot([x1, x2], [y1, y2], linestyle='--', color='skyblue', alpha = 0.3,  markersize=8, label='Ave SRCC without StairReward')
ax.text(0, 0.644, 0.664, va='center', ha='right', fontsize=10, color='skyblue', alpha = 0.6)


x1, y1 = 0, sum(data[1 : 6 : 2])/3
x2, y2 = 4.6, sum(data[1 : 6 : 2])/3
ax.plot([x1, x2], [y1, y2], linestyle='--', color='skyblue', alpha = 1,  markersize=8, label='Ave SRCC with StairReward')
ax.text(0, 0.6739, 0.6739, va='center', ha='right', fontsize=10, color='skyblue', alpha = 1)


x1, y1 = 5.3, sum(data[6 : 12 : 2])/3
x2, y2 = 10, sum(data[6 : 12 : 2])/3
ax.plot([x1, x2], [y1, y2], linestyle='--', color='lightcoral', alpha = 0.3,  markersize=8, label='Ave PLCC without StairReward')
ax.text(10, 0.7275, 0.7475, va='center', ha='left', fontsize=10, color='lightcoral', alpha = 0.3)

x1, y1 = 5.3, sum(data[7 : 12 : 2])/3
x2, y2 = 10, sum(data[7 : 12 : 2])/3
ax.plot([x1, x2], [y1, y2], linestyle='--', color='lightcoral', alpha = 1,  markersize=8, label='Ave PLCC with StairReward')
ax.text(10, 0.7658, 0.7858, va='center', ha='left', fontsize=10, color='lightcoral', alpha = 1)

ax.set_ylim(0, 0.88)
ax.set_xlim(0, 10)

ax2 = ax.twinx()
ax2.set_ylim(0, 0.88)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
ax.legend(loc='lower left')

plt.show()













# # 创建数据
# data_percetion = [0.7263, 0.7635, 0.4864, 0.5274, 0.4569, 0.5707, 0.6743, 0.7278]  # 这里的数据表示每个条的高度
# # data_percetion = [0.6674, 0.7252, 0.6789, 0.7463, 0.6831, 0.7576, 0.6372, 0.7258]
# # data_percetion = [0.7086, 0.7600, 0.6807, 0.7623, 0.5756, 0.6899, 0.7115, 0.7346]

# # data_align = [0.6914,	0.7134,	0.5357,	0.5350,	0.5024,	0.5486,	0.4210,	0.3851]
# # data_align = [0.6258,	0.6873,	0.6243,	0.7160,	0.6556,	0.7305,	0.4108	,0.5722]
# data_align = [0.6470,	0.7081,	0.6658,	0.7317,	0.5099,	0.5964,	0.6776,	0.6855]

# data = []

# for i in range(8):
#     data.append(data_align[i])

# fig, ax = plt.subplots(figsize=(10, 6))

# width = 0.5

# # 将第一和第二个条紧挨着，调整X坐标位置
# bars1 = ax.bar([1], data[0], width=width,  color='skyblue', alpha = 0.8)
# bars1 = ax.bar([1.5], data[1], width=width,  color='skyblue', alpha = 1)
# # bars1 = ax.bar([1, 1.5, 2.7, 3.2, 4,4.5][1 : 6 : 2], data[1 : 6 : 2], width=width,  color='skyblue', alpha = 1)

# bars2 = ax.bar([2.5], data[2], width=width,  color='lightcoral', alpha=0.8)
# bars2 = ax.bar([3], data[3], width=width,  color='lightcoral', alpha=1)


# bars3 = ax.bar([4], data[4], width=width,  color='turquoise', alpha=0.8)
# bars3 = ax.bar([4.5], data[5], width=width,  color='turquoise', alpha=1)


# bars4 = ax.bar([5.5], data[6], width=width,  color='lavender', alpha=0.8)
# bars4 = ax.bar([6], data[7], width=width,  color='lavender', alpha=1)



# # ax.set_xlabel('Groups')
# ax.set_ylabel('Values')
# # ax.set_title('SRCC and PLCC of Alignment on Model Quality Subsets')
# ax.set_title('SRCC and PLCC of Alignment on Style Subsets')
# ax.legend()

# y_offsets = [0.01] * 8

# x_positions = []
# for i in range(2):
#     x_positions.append(1 + 0.5 * i)

# for i in range(2):
#     x_positions.append(2.5 + 0.5 * i)

# for i in range(2):
#     x_positions.append(4 + 0.5 * i)

# for i in range(2):
#     x_positions.append(5.5 + 0.5 * i)
# for i in range(len(x_positions)):
#     ax.annotate(str(data[i]), (x_positions[i], data[i]), ha='center')



# # ax.set_xticks([1, 1.25, 1.5, 2.5, 2.75, 3, 4, 4.25, 4.5, 5.5, 5.75, 6])
# ax.set_xticks([1, 1.25, 1.5, 2.5, 2.75, 3, 4, 4.25, 4.5, 5.5, 5.75, 6])
# # ax.set_xticklabels(['SRCC', '\nAll','PLCC', 'SRCC','\nBad Models','PLCC', 'SRCC','\nMedium Models','PLCC', 'SRCC','\nGood Models','PLCC'])
# ax.set_xticklabels(['SRCC', '\nAbstract & Sci-fi','PLCC', 'SRCC','\nAnime & Realistic','PLCC', 'SRCC','\nBaroque','PLCC', 'SRCC','\nNo Style','PLCC'])

# ax.set_ylim(0, 0.87)

# ax.grid(True, axis='y', linestyle='--', alpha=0.3)

# # plt.axhline(y=data[0], color='skyblue', alpha = 0.8, linestyle='--')
# # plt.axhline(y=data[1], color='skyblue', alpha = 1, linestyle='--')
# plt.axhline(y=sum(data[0::2])/4, color='orange', alpha = 0.8, linestyle='--', label='Average SRCC')
# plt.axhline(y=sum(data[1::2])/4, color='blue', alpha = 0.8, linestyle='--', label='Average PLCC')
# ax.legend()
# plt.show()



# import numpy as np


# data = [0.6258,	0.6873,	0.6243,	0.7160,	0.6556,	0.7305,	0.4108	,0.5722]

# fig, ax = plt.subplots(figsize=(10, 6))

# # 分别取0,2,4,6索引和1,3,5,7索引的数据
# data1 = data[0::2]
# data2 = data[1::2]

# # X轴位置
# x_positions = [1.25, 2.75, 4.25, 5.75]

# # 绘制两个折线图
# plt.plot(x_positions, data1, marker='o', color='skyblue', linestyle='-', markersize=8, label='SRCC')
# plt.plot(x_positions, data2, marker='o', color='lightcoral', linestyle='-', markersize=8, label='PLCC')

# ax.set_ylabel('Values')
# ax.set_title('SRCC and PLCC of Alignment on Prompt Lengths Subsets')
# # ax.set_title('SRCC and PLCC of Perception on Prompt Lengths Subsets')
# ax.legend(['SRCC', 'PLCC'])

# # 标注数据点
# for i in range(len(x_positions)):
#     ax.annotate(f'{data1[i]:.4f}', (x_positions[i], data1[i] + 0.004), ha='center')
#     ax.annotate(f'{data2[i]:.4f}', (x_positions[i], data2[i] + 0.004), ha='center', va='bottom')
# # 找到最大值的索引
# max_index1 = np.argmax(data1)
# max_index2 = np.argmax(data2)

# # 在最大值上标注星星
# ax.plot(x_positions[max_index1], data1[max_index1], marker='*', markersize=16, color='skyblue')
# ax.plot(x_positions[max_index2], data2[max_index2], marker='*', markersize=16, color='lightcoral')
# ax.set_xticks(x_positions)
# ax.set_xticklabels(['Prompt 0', 'Prompt 1', 'Prompt 2', 'Prompt 3'])

# ax.set_ylim(0.4, 0.75)

# ax.grid(True, axis='y', linestyle='--', alpha=0.3)

# plt.show()























# # 创建数据
# data_percetion = [0.7263, 0.7635, 0.4864, 0.5274, 0.4569, 0.5707, 0.6743, 0.7278]  # 这里的数据表示每个条的高度
# # data_percetion = [0.6674, 0.7252, 0.6789, 0.7463, 0.6831, 0.7576, 0.6372, 0.7258]
# # data_percetion = [0.7086, 0.7600, 0.6807, 0.7623, 0.5756, 0.6899, 0.7115, 0.7346]

# data_align = [0.6914,	0.7134,	0.5357,	0.5350,	0.5024,	0.5486,	0.4210,	0.3851]
# # data_align = [0.6674, 0.7252, 0.6789, 0.7463, 0.6831, 0.7576, 0.6372, 0.7258]
# # data_align = [0.7086, 0.7600, 0.6807, 0.7623, 0.5756, 0.6899, 0.7115, 0.7346]

# data = []

# for i in range(8):
#     data.append(data_percetion[i])
#     data.append(data_align[i])

# fig, ax = plt.subplots(figsize=(10, 6))

# width = 0.3

# # 将第一和第二个条紧挨着，调整X坐标位置
# bars1 = ax.bar([1], data[0], width=width,  color='skyblue', alpha = 0.5)
# bars1 = ax.bar([1.3], data[1], width=width,  color='skyblue', alpha = 0.65)
# bars1 = ax.bar([1.6], data[2], width=width,  color='skyblue', alpha = 0.8)
# bars1 = ax.bar([1.9], data[3], width=width,  color='skyblue', alpha = 1)
# # bars1 = ax.bar([1, 1.5, 2.7, 3.2, 4,4.5][1 : 6 : 2], data[1 : 6 : 2], width=width,  color='skyblue', alpha = 1)
# bars2 = ax.bar([2.9], data[4], width=width,  color='lightcoral', alpha=0.5)
# bars2 = ax.bar([3.2], data[5], width=width,  color='lightcoral', alpha=0.65)
# bars2 = ax.bar([3.5], data[6], width=width,  color='lightcoral', alpha=0.8)
# bars2 = ax.bar([3.8], data[7], width=width,  color='lightcoral', alpha=1)

# bars3 = ax.bar([4.8], data[8], width=width,  color='turquoise', alpha=0.5)
# bars3 = ax.bar([5.1], data[9], width=width,  color='turquoise', alpha=0.65)
# bars3 = ax.bar([5.4], data[10], width=width,  color='turquoise', alpha=0.8)
# bars3 = ax.bar([5.7], data[11], width=width,  color='turquoise', alpha=1)

# bars4 = ax.bar([6.7], data[12], width=width,  color='lavender', alpha=0.5)
# bars4 = ax.bar([7], data[13], width=width,  color='lavender', alpha=0.65)
# bars4 = ax.bar([7.3], data[14], width=width,  color='lavender', alpha=0.8)
# bars4 = ax.bar([7.6], data[15], width=width,  color='lavender', alpha=1)



# # ax.set_xlabel('Groups')
# ax.set_ylabel('Values')
# # ax.set_title('SRCC and PLCC of Perception on Model Quality Subsets')
# ax.set_title('SRCC and PLCC of Perception on Style Subsets')
# ax.legend()

# y_offsets = [0.01] * 16

# x_positions = []
# for i in range(4):
#     x_positions.append(1 + 0.3 * i)

# for i in range(4):
#     x_positions.append(2.9 + 0.3 * i)

# for i in range(4):
#     x_positions.append(4.8 + 0.3 * i)

# for i in range(4):
#     x_positions.append(6.7 + 0.3 * i)
# for i in range(len(x_positions)):
#     ax.annotate(str(data[i]), (x_positions[i], data[i]), ha='center')

# ax.set_xticks([1.45, 3.35, 5.25, 7.15, 1.15, 1.75, 3.05, 3.65, 4.95, 5.55, 6.85, 7.45])
# ax.set_xticklabels(['\nAll', '\nBad Models', '\nMedium Models', '\nGood Models', 'SRCC','PLCC', 'SRCC','PLCC', 'SRCC','PLCC', 'SRCC','PLCC'])
# # ax.set_xticks([1.15, 1.75])
# # ax.set_xticklabels(['SRCC','PLCC'])

# # ax.set_xticks([1, 1.25, 1.5, 2.5, 2.75, 3, 4, 4.25, 4.5, 5.5, 5.75, 6])
# # ax.set_xticks([1, 1.25, 1.5, 2.5, 2.75, 3, 4, 4.25, 4.5, 5.5, 5.75, 6])
# # ax.set_xticklabels(['SRCC', '\nAll','PLCC', 'SRCC','\nBad Models','PLCC', 'SRCC','\nMedium Models','PLCC', 'SRCC','\nGood Models','PLCC'])
# # ax.set_xticklabels(['SRCC', '\nAbstract & Sci-fi','PLCC', 'SRCC','\nAnime & Realistic','PLCC', 'SRCC','\nBaroque','PLCC', 'SRCC','\nNo Style','PLCC'])

# ax.set_ylim(0, 0.87)

# ax.grid(True, axis='y', linestyle='--', alpha=0.3)


# # plt.axhline(y=sum(data[0::2])/4, color='orange', alpha = 0.8, linestyle='--', label='Average SRCC')
# # plt.axhline(y=sum(data[1::2])/4, color='blue', alpha = 0.8, linestyle='--', label='Average PLCC')
# ax.legend()
# plt.show()
