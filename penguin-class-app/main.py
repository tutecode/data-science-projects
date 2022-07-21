import pandas as pd
import numpy as np


# read data from csv with pd
url_adelie = "https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-pal.219.3&entityid=002f3893385f710df69eeebe893144ff"
url_gentoo = "https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-pal.220.3&entityid=e03b43c924f226486f2f0ab6709d2381"
url_chinstrap = "https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-pal.221.2&entityid=fe853aa8f7a59aa84cdd3197619ef462"

df_adelie = pd.read_csv(url_adelie, index_col=False)
df_gentoo = pd.read_csv(url_gentoo, index_col=False)
df_chinstrap = pd.read_csv(url_chinstrap, index_col=False)

# merge data
df_penguins = pd.concat([df_adelie, df_gentoo, df_chinstrap])

# clean data
df_penguins.drop(['studyName', 'Sample Number', 'Region',
                  'Stage', 'Individual ID', 'Clutch Completion', 'Date Egg',
                  'Delta 15 N (o/oo)', 'Delta 13 C (o/oo)', 'Comments'],
                 axis=1, inplace=True)


df_penguins.columns = ['species', 'island', 'bill_length_mm',
                       'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']

# replace name
df_penguins['species'].mask(df_penguins['species'] == 'Adelie Penguin (Pygoscelis adeliae)', 'Adelie', inplace=True)
df_penguins['species'].mask(df_penguins['species'] == 'Gentoo penguin (Pygoscelis papua)', 'Gentoo', inplace=True)
df_penguins['species'].mask(df_penguins['species'] == 'Chinstrap penguin (Pygoscelis antarctica)', 'Chinstrap', inplace=True)

# drop 'NaN' rows
df_penguins = df_penguins.replace('', np.nan)
df_penguins = df_penguins.dropna(axis=0, how='any')

# str to_lowecase
df_penguins['sex'] = df_penguins['sex'].apply(str.lower)

# df to csv
df_penguins = df_penguins.to_csv('Penguins.csv', encoding='utf-8', index=False)

# df_penguins = df_penguins.fillna("", inplace=False)
df_penguins = pd.read_csv("Penguins.csv", index_col=False)

print(df_penguins)
