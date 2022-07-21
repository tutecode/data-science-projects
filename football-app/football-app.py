import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title('NFL Football Stats (Rushing) Explorer')

st.markdown("""
This app performs simple webscraping of NFL Football player stats data (focusing on Rushing)!
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Data source: ** [pro-football-reference.com](https://www.pro-football-reference.com/).
""")

st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1990,2020))))

# Web scraping of NFL player stats
# https://www.pro-football-reference.com/years/2019/rushing.htm


@st.cache
def load_data(year):
    url = "https://www.pro-football-reference.com/years/" + str(year) + "/rushing.htm"
    html = pd.read_html(url, header=1)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index)  # Deletes repeating headers in content
    raw = raw.fillna(0)
    playerstats = raw.drop(['Rk'], axis=1)

    return playerstats


playerstats = load_data(selected_year)

# Sidebar - Team selection
sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)

# Sidebar - Position selection
unique_pos = ['RB', 'QB', 'WR', 'FB', 'TE']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

# Filtering data
df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

# Main Page
st.header('Display Player Stats of Selected Team(s)')
if st.button('Glosary'):
    st.write('''
        **Rk -- Rank**\n
            This is a count of the rows from top to bottom.\n
            It is recalculated following the sorting of a column.\n
        **Age** -- Player's age on December 31st of that year\n
        **Pos -- Position**\n
            In player and team season stats,\n
            Capitals indicates primary starter.\n
            Lower-case means part-time starter.\n
        **Games**\n
            **G** -- Games played\n
            **GS** -- Games started as an offensive or defensive player\n
        **Rushing**\n
            **Att** -- Rushing Attempts (sacks not included in NFL)\n
            **Yds** -- Rushing Yards Gained (sack yardage is not included by NFL)\n
            **TD** -- Rushing Touchdowns\n
            **1D** -- First downs rushing\n
            **Lng** -- Longest Rushing Attempt\n
            **Y/A** -- Rushing Yards per Attempt\n
            Minimum 6.25 rushes per game scheduled to qualify as leader.\n
            Minimum 750 rushes to qualify as career leader.\n
            **Y/G** -- Rushing Yards per Game\n
            (minimum half a game per game scheduled to qualify as leader)\n
            (Rushing Yards)/(Games Played)\n
        **Fumbles**\n
            **Fmb** -- Number of times fumbled both lost and recovered by own team\n
            These represent ALL fumbles by the player on offense, defense, and special teams.\n
            Available for player games since 1990.
    ''')
st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
st.dataframe(df_selected_team)

# Download Football player stats data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806


def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'

    return href


st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

# Heatmap
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    df_selected_team.to_csv('output.csv', index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot(fig)
