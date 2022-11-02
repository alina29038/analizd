import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import RadioButtons
from sklearn.linear_model import LinearRegression

model = LinearRegression()
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['lines.linestyle'] = '-'
matplotlib.rcParams.update({'font.size': 7})
plt.style.use('bmh')
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

exl = pd.read_excel('sheet.xlsx',index_col=0, na_filter=1)
fig, ax = plt.subplots(1, figsize=(13,7))
plt.subplots_adjust(left=0.3)
exlData = pd.DataFrame(exl)

x = ('январь',
    'февраль',
    'март',
    'апрель',
    'май',
    'июнь',
    'июль',
    'август',
    'сентябрь',
    'октябрь',
    'ноябрь',
    'декабрь'
            )

exlData = exlData.fillna(0)

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)
pd.options.mode.chained_assignment = None


aprel2017 = exlData.iloc[3,1] / exlData.iloc[2,1] * exlData.iloc[2,0]
exlData.iloc[3,0] = aprel2017
iun2018 = (exlData.iloc[5,2]/ exlData.iloc[4,2] * exlData.iloc[4,1]).round(0)
exlData.iloc[5,1] = iun2018
noiabr2019 = (exlData.iloc[10,3] / exlData.iloc[9,3] * exlData.iloc[9,2]).round(0)
exlData.iloc[10,2] = noiabr2019
fevral2020 = (exlData.iloc[1,4] / exlData.iloc[0,4] * exlData.iloc[0,3]).round(0)
exlData.iloc[1,3] = fevral2020
avgust2020 = (exlData.iloc[7,4] / exlData.iloc[6,4] * exlData.iloc[6,3]).round(0)
exlData.iloc[7,3] = avgust2020

month = 0
for i in range(12):
    middleX = (2017 + 2018 + 2019 + 2020 + 2021 + 2022) / 6
    middleY = (exlData.iloc[month, :].values[0] + exlData.iloc[month, :].values[1] + exlData.iloc[month, :].values[2]+ exlData.iloc[month, :].values[3] + exlData.iloc[month, :].values[4] + exlData.iloc[month, :].values[5]) / 6
    b = (((2017 - middleX) * (exlData.iloc[month, :].values[0] - middleY)) + ((2018 - middleX) * (exlData.iloc[month, :].values[1] - middleY)) + ((2019 - middleX) * (exlData.iloc[month, :].values[2] - middleY)) +((2020 - middleX) * (exlData.iloc[month, :].values[3] - middleY)) + ((2021 - middleX) * (exlData.iloc[month, :].values[4] - middleY) +(2022 - middleX) * (exlData.iloc[month, :].values[5] - middleY)))/ (((2017 - middleX)**2) + ((2018 - middleX)**2) + ((2019 - middleX)**2) + ((2020 - middleX)**2) + ((2021 - middleX)**2) + ((2022 - middleX)**2))
    a = middleY - b * middleX
    res = a + b * 2023
    res = round(res,0)
    exlData.iloc[:, 6].values[month] = res
    month += 1

month = 0
for i in range(0,12):
    middleX = (2017 + 2018 + 2019 + 2020 + 2021 + 2022 + 2023) / 7
    middleY = (exlData.iloc[month, :].values[0] + exlData.iloc[month, :].values[1] + exlData.iloc[month, :].values[2] +
    exlData.iloc[month, :].values[3] + exlData.iloc[month, :].values[4] + exlData.iloc[month, :].values[5] + exlData.iloc[month, :].values[6])/ 7
    b = (((2017 - middleX) * (exlData.iloc[month, :].values[0] - middleY)) + ((2018 - middleX) * (exlData.iloc[month, :].values[1] - middleY)) +
         ((2019 - middleX) * (exlData.iloc[month, :].values[2] - middleY)) + ((2020 - middleX) * (exlData.iloc[month, :].values[3] - middleY)) +
         ((2021 - middleX) * (exlData.iloc[month, :].values[4] - middleY) + (2022 - middleX) * (exlData.iloc[month, :].values[5] - middleY)) +
         (2023 - middleX) * (exlData.iloc[month, :].values[6] - middleY)) \
        / (((2017 - middleX)) ** 2 + ((2018 - middleX) ** 2) + ((2019 - middleX) ** 2) + ((2020 - middleX) ** 2) + ((2021 - middleX) ** 2) + ((2022 - middleX) ** 2) +
           ((2023 - middleX) ** 2))
    a = middleY - b * middleX
    res = a + b * 2024
    res = round(res,0)
    exlData.iloc[:, 7].values[month] = res
    month += 1

exl2 = pd.read_excel('sheet2.xlsx',index_col=0, na_filter=1)
exl2Data = pd.DataFrame(exl2)
exl2Data = exl2Data.fillna(0)

k = 0
for i in range(12):
    z = 1.96
    middleX = ((exlData.iloc[0,6]+exlData.iloc[1,6]+exlData.iloc[2,6]+exlData.iloc[3,6]+exlData.iloc[4,6]+exlData.iloc[5,6]+exlData.iloc[6,6]+exlData.iloc[7,6]+exlData.iloc[8,6]+exlData.iloc[9,6]+exlData.iloc[10,6]+exlData.iloc[11,6])/12).round(0)
    stDev = ((((exlData.iloc[0,6] - middleX)**2 + (exlData.iloc[1,6] - middleX)**2 +(exlData.iloc[2,6] - middleX)**2 +(exlData.iloc[3,6] - middleX)**2 +(exlData.iloc[4,6] - middleX)**2 +(exlData.iloc[5,6] - middleX)**2 +(exlData.iloc[6,6] - middleX)**2 +(exlData.iloc[7,6] - middleX)**2 + (exlData.iloc[8,6] - middleX)**2 + (exlData.iloc[9,6] - middleX)**2 +(exlData.iloc[10,6] - middleX)**2 +(exlData.iloc[11,6] - middleX)**2)/11)**(1/2)).round(0)
    deviation = (z * (stDev/(12)**(1/2))).round(0)
    exl2Data.iloc[k,0] = exlData.iloc[k,6] + deviation
    exl2Data.iloc[k,1] = exlData.iloc[k,6] - deviation
    k+=1

k = 0
for i in range(12):
    z = 1.96
    middleX = ((exlData.iloc[0,7]+exlData.iloc[1,7]+exlData.iloc[2,7]+exlData.iloc[3,7]+exlData.iloc[4,7]+exlData.iloc[5,7]+exlData.iloc[6,7]+exlData.iloc[7,7]+exlData.iloc[8,7]+exlData.iloc[9,7]+exlData.iloc[10,7]+exlData.iloc[11,7])/12).round(0)
    stDev = ((((exlData.iloc[0,7] - middleX)**2 + (exlData.iloc[1,7] - middleX)**2 +(exlData.iloc[2,7] - middleX)**2 +(exlData.iloc[3,7] - middleX)**2 +(exlData.iloc[4,7] -middleX)**2 +(exlData.iloc[5,7] - middleX)**2 +(exlData.iloc[6,7] - middleX)**2 +(exlData.iloc[7,7] - middleX)**2 + (exlData.iloc[8,7] - middleX)**2 + (exlData.iloc[9,7] - middleX)**2 +(exlData.iloc[10,7] - middleX)**2 +(exlData.iloc[11,7] - middleX)**2)/11)**(1/2)).round(0)
    deviation = (z * (stDev/(12)**(1/2))).round(0)
    exl2Data.iloc[k,2] = exlData.iloc[k,7] + deviation
    exl2Data.iloc[k,3] = exlData.iloc[k,7] - deviation
    k+=1

s1 = exlData.iloc[:, 0].values
s2 = exlData.iloc[:, 1].values
s3 = exlData.iloc[:, 2].values
s4 = exlData.iloc[:, 3].values
s5 = exlData.iloc[:, 4].values
s6 = exlData.iloc[:, 5].values
s7 = exlData.iloc[:, 6].values
s8 = exlData.iloc[:, 7].values
lr = exlData.iloc[[11, 0, 1], 0:6].values
lr2 = exlData.iloc[2:5, 0:6].values
lr3 = exlData.iloc[5:8, 0:6].values
lr4 = exlData.iloc[8:11, 0:6].values


def click(label):
    ax.clear()
    if label == "2017":
        ax.plot(x, s1, lw=2, color='dodgerblue')
        ax.set_title("2017", c='dodgerblue')
    elif label == "Все":
        ax.set_title("2017-2023")
        ax.plot(exlData)
    elif label == "2018":
        ax.plot(x, s2, lw=2, color='firebrick')
        ax.set_title("2018", c='firebrick')
    elif label == "2019":
        ax.plot(x, s3, lw=2, color='mediumpurple')
        ax.set_title("2019", c='mediumpurple')
    elif label == "2020":
        ax.plot(x, s4, lw=2, color='seagreen')
        ax.set_title("2020", c='seagreen')
    elif label == "2021":
        ax.plot(x, s5, lw=2, color='coral')
        ax.set_title("2021", c='coral')
    elif label == "2022":
        ax.plot(x, s6, lw=2, color='lightpink')
        ax.set_title("2022", c='lightpink')
    elif label == "2023":
        ax.plot(x, s7, lw=2, color='aqua')
        ax.set_title("2023", c='aqua')
    elif label == "2024":
        ax.plot(x, s8, lw=2, color='mediumspringgreen')
        ax.set_title("2024", c='mediumspringgreen')

    elif label == "Linear Regression SEAS1":
        mon = np.array([[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3]])
        ax.scatter(mon, lr, color='gray')
        ax.plot([mon.min(), mon.max()], [lr.min(), lr.max()], 'k--')
        ax.set_title("Linear Regression SEAS1", c='red')

    elif label == "Linear Regression SEAS2":
        mon = np.array([[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3]])
        ax.scatter(mon, lr2,color='gray')
        ax.plot([mon.min(), mon.max()],[lr2.min(), lr2.max()], 'k--')
        # ax.plot([['mar'], ['apr'], ['may']], y_pred, lw=2, color='red')
        ax.set_title("Linear Regression SEAS2", c='red')

    elif label == "Linear Regression SEAS3":
        mon = np.array([[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3]])
        ax.scatter(mon, lr3, color='gray')
        ax.plot([mon.min(), mon.max()], [lr3.min(), lr3.max()], 'k--')
        ax.set_title("Linear Regression SEAS3", c='red')

    elif label == "Linear Regression SEAS4":
        mon = np.array([[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3]])
        ax.scatter(mon, lr4, color='gray')
        ax.plot([mon.min(), mon.max()], [lr4.min(), lr4.max()], 'k--')
        ax.set_title("Linear Regression SEAS4", c='red')
    plt.draw()

print(exlData)
print('=================================================')
print(exl2Data)



rax = plt.axes([0.02, 0.55, 0.20, 0.35], facecolor='white')
radio = RadioButtons(rax, ('Все', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', 'Linear Regression SEAS1', 'Linear Regression SEAS2', 'Linear Regression SEAS3', 'Linear Regression SEAS4'), activecolor='k')
plt.title(r'Доход магазинчкика')


plt.figtext(0.93, 0.65, '2024', size=12, c='mediumspringgreen')
plt.figtext(0.93, 0.60, '2023', size=12, c='aqua')
plt.figtext(0.93, 0.55, '2022', size=12, c='lightpink')
plt.figtext(0.93, 0.50, '2021', size=12, c='coral')
plt.figtext(0.93, 0.45, '2020', size=12, c='seagreen')
plt.figtext(0.93, 0.40, '2019', size=12, c='mediumpurple')
plt.figtext(0.93, 0.35, '2018', size=12, c='firebrick')
plt.figtext(0.93, 0.30, '2017', size=12, c='dodgerblue')

ax.plot(exlData)
ax.set_title("2017-2024")
radio.on_clicked(click)

plt.show()