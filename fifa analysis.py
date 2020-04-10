# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 转换金钱格式
def transfer_money(money):
    if isinstance(money, str) is False:
        return money
    if(money[-1] == '0'):
        number = float(money[1:])
    else:
        number = float(money[1:-1]) * (1000000 if money[-1] == 'M' else 1000)
    return number

# 读入文件
fifa_data = pd.read_csv('data.csv', index_col=0)
fifa_data = fifa_data.drop(columns='Jersey Number')

# 转换表格value和wage格式
transferred_values = list(map(lambda v: transfer_money(v), list(fifa_data['Value'].values)))
fifa_data['Value'] = transferred_values

transferred_wages = list(map(lambda v: transfer_money(v), list(fifa_data['Wage'].values)))
fifa_data['Wage'] = transferred_wages

transferred_RC = list(map(lambda v: transfer_money(v), list(fifa_data['Release Clause'].values)))
fifa_data['Release Clause'] = transferred_RC
fifa_data['Relase Clause'] = fifa_data['Release Clause'].fillna(fifa_data[pd.notna(fifa_data['Release Clause'])]['Release Clause'].mean())

# 俱乐部总价值计算
club_value = fifa_data.groupby('Club').sum()['Value'].to_frame()
club_value = club_value.sort_values(by='Value', ascending=False)
# 俱乐部总薪酬计算
club_wage = fifa_data.groupby('Club').sum()['Wage'].to_frame()
club_wage = club_wage.sort_values(by='Wage', ascending=False)

# 弱脚计数
weak_foot_data = fifa_data.groupby('Weak Foot').count()['ID'].to_frame()
weak_foot_data.columns = ['Count']
# 前10value和前10wage的俱乐部
top10clubs_value = club_value.iloc[0:10, :]
top10clubs_wage = club_wage.iloc[0:10, :]

top10clubs = list(top10clubs_value.index)

#################################### 1. 前10俱乐部 前10国籍数 ####################################
# 获得前10俱乐部球员信息
top10clubs_players_info = []
for club in top10clubs:
    club_players_info = fifa_data[fifa_data['Club'] == club]
    top10clubs_players_info.append(club_players_info)

top10clubs_players_info = pd.concat(top10clubs_players_info)
# 获得是前10俱乐部的并且球员国籍数目前10的球员信息
top10clubs_countries = top10clubs_players_info.groupby('Nationality').count()['ID'].copy().to_frame()
top10clubs_countries.columns = ['Count']
top10clubs_countries.sort_values(ascending=False, by='Count', inplace=True)
top10clubs_countries = top10clubs_countries.index

top10_countries_info = []
for country in top10clubs_countries:
    country_info = top10clubs_players_info[top10clubs_players_info['Nationality'] == country]
    top10_countries_info.append(country_info)
top10_countries_info = pd.concat(top10_countries_info)

# 画出前10俱乐部的前10国籍数的bar chart
plt.style.use('dark_background')
plt.figure(figsize=(10,7))
sns.countplot(top10_countries_info['Nationality'], color='orange')
plt.title('Top 10 clubs countries source - by number', fontsize=20)
plt.xticks(rotation=-90)
plt.show()

# pie chart
'''countries_per = top10_countries_info.groupby('Nationality').count()['ID'].to_frame()
countries_per.columns = ['Count']
player_countries = list(countries_per.index)
countries_per = list(100 * countries_per.values.flatten() / np.sum(countries_per.values.flatten()))
countries_per = [float('{:.2f}'.format(p)) for p in countries_per]

plt.pie(countries_per, labels=player_countries, autopct='%.2f%%', shadow=True, explode=[0.3]*len(player_countries))
plt.show()'''

######################################## 2. 运动选择的脚 ########################################

sns.countplot(fifa_data['Preferred Foot'], palette='pink')
plt.title('Most Preferred Foot of the players')
plt.show()

######################################### 3. 运动弱的脚 #########################################

weak_foots = list(weak_foot_data.index)
weak_foots_count = [weak_foot_data[weak_foot_data.index==index]['Count'].values[0] for index in weak_foots]
weak_foots_count = [float('{:.2f}'.format(count / sum(weak_foots_count) * 100.0)) for count in weak_foots_count]
plt.title('Weak foot')
plt.pie(weak_foots_count, labels=weak_foots,autopct='%.2f%%', shadow=True, explode=[0.1]*len(weak_foots))
plt.show()

######################################## 4. 球员位置分布 ########################################
def position_map(position):
    if(isinstance(position, str) is False):
        return position
    positions_dict = {'GK':'Goalkeeper', 'ST':'Striker', 'F':'Forward', 'A':'Attack',\
                  'B':'Back', 'W':'Wing', 'D':'Defender', 'M':'Midfield', \
                  'S':'Side', 'L':'Left', 'R':'Right', 'C':'Center'}
    if(position in positions_dict.keys()):
        transferred_position = positions_dict[position]
    else:
        transferred_position = ' '.join([positions_dict[c] for c in position])
    
    return transferred_position


positions_data = fifa_data['Position'].copy()
positions_data = positions_data.apply(position_map)

plt.figure(figsize = (10, 6))
plt.xticks(rotation=-90)
sns.countplot(positions_data, palette='bone')

###################################### 5. 球员体重身高分析 ######################################

weights_data = fifa_data['Weight'].copy().to_frame()
weights_data[pd.notna(weights_data['Weight'])] = weights_data[pd.notna(weights_data['Weight'])].applymap(lambda w:float(w.replace('lbs', '')))
mean_weight = weights_data[pd.notna(weights_data['Weight'])].mean()
weights_data = weights_data['Weight'].to_frame().fillna(mean_weight)
fifa_data['Weight'] = weights_data['Weight']

heights_data = fifa_data['Height'].to_frame()
heights_data[pd.notna(heights_data['Height'])] = heights_data[pd.notna(heights_data['Height'])].applymap(lambda w:float(w.replace('\'', '.')))
mean_height = heights_data[pd.notna(heights_data['Height'])].mean()
heights_data = heights_data['Height'].to_frame().fillna(mean_height)
fifa_data['Height'] = heights_data['Height']

plt.scatter(fifa_data['Weight'].values, fifa_data['Height'].values)

def quantile(df, col_name):
    minimum = df[col_name].min()
    q1 = df[col_name].quantile(0.25)
    q2 = df[col_name].quantile(0.5)
    q3 = df[col_name].quantile(0.75)
    maximum = df[col_name].max()+0.001
    count = []
    qs = [minimum, q1, q2, q3, maximum]
    
    qrange = [str(qs[i])+'-'+str(qs[i+1])[:5] for i in range(len(qs)-1)]
    
    
    for i in range(len(qs)-1):
        count.append(df[(df[col_name] >= qs[i]) & (df[col_name] < qs[i+1])][col_name].count())
    
    return qrange, count

# Weight
weight_qrange, weight_count = quantile(df=fifa_data, col_name='Weight')
sns.set(style="whitegrid")
sns.barplot(weight_qrange, weight_count, palette="Blues_d")
plt.title('Weight Count')
plt.xlabel('Weight range')
plt.ylabel('Count')
plt.show()

# Height
height_qrange, height_count = quantile(df=fifa_data, col_name='Height')
sns.set(style="whitegrid")
sns.barplot(height_qrange, height_count, palette="Blues_d")
plt.title('Height Count')
plt.xlabel('Height range')
plt.ylabel('Count')
plt.show()

# Weight distribution
plt.title('Distribution of weight of players', fontsize=20)
plt.xlabel('Weight range', fontsize=16)
plt.ylabel('Count', fontsize=16)
sns.distplot(fifa_data['Weight'])
plt.show()

# Height distribution
plt.title('Distribution of height of players', fontsize=20)
plt.xlabel('Height range', fontsize=16)
plt.ylabel('Count', fontsize=16)
sns.distplot(fifa_data['Height'])
plt.show()
####################################### 6. 薪酬分布分析 #######################################
plt.figure(figsize = (10, 5))
plt.xlabel('Wage range for players', fontsize=16)
plt.ylabel('Count of the players', fontsize=16)
plt.title('Distribution of wages of players', fontsize=20)
sns.distplot(fifa_data['Wage'], color='Blue')
plt.show()

######################################## 7. Score分析 ########################################
plt.title('Histogram of Speciality Score', fontsize=20)
plt.xlabel('Speciality Score')
plt.ylabel('Count')
sns.distplot(fifa_data['Special'], bins=58, kde=False, color='r')
plt.show()
plt.title('Histogram of Overall Score', fontsize=20)
plt.xlabel('Overall Score')
sns.distplot(fifa_data['Overall'], bins=58, kde=False, color='m')
plt.show()
sns.distplot(fifa_data['Potential'], bins=58, kde=False, color='y')
plt.show()
plt.xlabel('Overall')
plt.ylabel('Potential')
fifa_data['Potential Improvement'] = fifa_data['Potential'].to_frame().values.flatten() - fifa_data['Overall'].to_frame().values.flatten()

unique_overall_scores = list(set(fifa_data['Overall'].values))
max_improvement = [fifa_data[fifa_data['Overall'] == s]['Potential Improvement'].max() for s in unique_overall_scores]

plt.title('Potential improvement of players', fontsize=20)
plt.xlabel('Overall Score', fontsize=16)
plt.ylabel('Potential Improvement', fontsize=16)
plt.ylim((0, max(max_improvement) + np.std(max_improvement) / 6))
plt.fill_between(unique_overall_scores, [0]*len(max_improvement), max_improvement, alpha=0.6)


plt.show()
######################### 8. Overall Score & Age wrt Preferred foot #########################
plt.figure(figsize=(20, 7))
plt.title('Comparison of Overall Scrore and age wrt Preferred foot', fontsize=20)
plt.style.use('classic')
sns.boxenplot(fifa_data['Overall'], fifa_data['Age'] ,hue=fifa_data['Preferred Foot'], palette='Blues_d')
plt.show()

######################## 9. Overall Score wrt Intenation Reputation ########################
plt.style.use('classic')
plt.title('Ratings vs Reputation', fontsize=20)
plt.xlabel('Overall Ratings', fontsize=16)
plt.ylabel('International Reputation', fontsize=16)
plt.scatter(fifa_data['Overall'], fifa_data['International Reputation'], s=fifa_data['Age']*20, c='pink')
plt.show()

################################## 9. Correlation heatmap ##################################
selected_cols = ['Age', 'Nationality', 'Overall', 'Potential', 'Club',
                 'Value', 'Wage', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot',
                 'Skill Moves', 'Work Rate', 'Body Type','Position', 
                 'Height', 'Weight','Finishing', 'HeadingAccuracy',
                 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',
                 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility',
                   'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength',
                   'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision',
                   'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle',
                   'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes',
                   'Release Clause']
plt.style.use('ggplot')
plt.figure(figsize=(30,20))
sns.heatmap(fifa_data[selected_cols].corr(), annot=True)

###################### 10. Different Nations Participating in FIFA 2019 ######################

plt.style.use('dark_background')
plt.title('Different Nations Participating in FIFA 2019', fontsize=30, fontweight=20)
plt.xlabel('Name of The Country')
plt.ylabel('Count')
fifa_data['Nationality'].value_counts().head(80).plot.bar(color='orange', figsize=(20,7))

############### 11. Distribution of Weights of players from different countries ###############
some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia')
data_countries = fifa_data.loc[fifa_data['Nationality'].isin(some_countries)]

plt.figure(figsize=(15, 7))
plt.title('Distribution of Weights of players from different countries', fontsize=30, fontweight=20)
sns.violinplot(x=data_countries['Nationality'], y=data_countries['Weight'], palette='Reds')

plt.show()

############################ 12. Defining the features of players ###########################
player_features = ('Acceleration', 'Aggression', 'Agility', 
                   'Balance', 'BallControl', 'Composure', 
                   'Crossing', 'Dribbling', 'FKAccuracy', 
                   'Finishing', 'GKDiving', 'GKHandling', 
                   'GKKicking', 'GKPositioning', 'GKReflexes', 
                   'HeadingAccuracy', 'Interceptions', 'Jumping', 
                   'LongPassing', 'LongShots', 'Marking', 'Penalties')

position_feature_mean = fifa_data.groupby(fifa_data['Position'])[player_features].mean()

for i in range(position_feature_mean.shape[0]):
    print('Position '+ position_feature_mean.index[i] + ': ' + ', '.join(list(position_feature_mean.iloc[i,:].nlargest(5).index)))    

plt.style.use('ggplot')
plt.figure(figsize=(15, 30))
for i in range(position_feature_mean.shape[0]):
    top_features = dict(position_feature_mean.iloc[i,:].nlargest(5))
    categories = top_features.keys()
    N = len(categories)
    values = list(top_features.values())
    values+=values[:1]
    angles = [n/float(N) * 2 * np.pi for n in range(N)]
    angles+=angles[:1]
    plt.subplot(9,3,i+1,polar=True)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.rcParams['axes.titlepad'] = 10
    plt.title(position_feature_mean.index[i], fontsize=12, fontweight=30, color='black')
    
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    plt.yticks([25,50,75], ['25', '50', '75'], color='grey', size=10)
    plt.ylim(0,100)
    #
    plt.plot(angles, values, linewidth=1, linestyle='solid', color='b')
    plt.fill(angles, values, 'b', alpha=0.1)

plt.show()
##################### 12. Relationship btwn Dribbling and BallControl ####################

plt.style.use('dark_background')
lm = sns.lmplot(x='BallControl', y='Dribbling', data=fifa_data, col='Preferred Foot', scatter_kws={'color':'red', 'alpha':0.1}, line_kws={'color':'red'})
fig = lm.fig 
plt.show()

########################## 13. Relationship btwn Raring and Age #########################

plt.style.use('classic')
sns.lineplot(fifa_data['Age'], fifa_data['Overall'], palette = 'Wistia')
plt.title('Age vs Overall Score', fontsize = 20)

plt.show()


############################## 13. Country vs Overall Score #############################

import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected = True)

import plotly.graph_objs as go


country_rating = fifa_data.groupby('Nationality')['Overall'].sum().reset_index()

trace = [go.Choropleth(colorscale='YlOrRd',
                       locationmode='country names',
                       locations=country_rating['Nationality'],
                       text=country_rating['Nationality'],
                       z=country_rating['Overall'])]
layout = go.Layout(title='Country vs Overall Score')
fig = go.Figure(data=trace, layout=layout)
py.plot(fig, filename='CountryvsOverallScore.html')
