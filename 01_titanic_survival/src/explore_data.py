import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    df = pd.read_csv('https://hbiostat.org/data/repo/titanic3.csv')
    df.to_csv(r'c:\Users\jhsim\Erica261\M.L\projects\01_titanic_survival\data\titanic5.csv', index=False)
    print('Downloaded Titanic5 OK')
except Exception as e:
    print(f'Download failed: {e}, using local file')
    df = pd.read_csv(r'c:\Users\jhsim\Erica261\M.L\projects\01_titanic_survival\data\raw_titanic.csv')

df.columns = df.columns.str.lower().str.replace('.', '_', regex=False)
print('Columns:', list(df.columns))
print('Shape:', df.shape)
print()

df['Title'] = df['name'].str.extract(r', (\w+)\.')
df['Surname'] = df['name'].str.split(',').str[0].str.strip()

print('Male Titles:')
print(df[df['sex']=='male']['Title'].value_counts())
print()
print('Female Titles:')
print(df[df['sex']=='female']['Title'].value_counts())
print()

# Surname overlap (Mr & Mrs travelling together)
mr_surnames = set(df[df['Title']=='Mr']['Surname'])
mrs_surnames = set(df[df['Title']=='Mrs']['Surname'])
overlap = mr_surnames & mrs_surnames
print(f'Surnames with BOTH Mr and Mrs on board: {len(overlap)} / {len(mr_surnames)} Mr-surnames')
print(f'  => These are likely married couples travelling together')
print()

# Mr stats
mr = df[df['Title'] == 'Mr']
print(f'Mr passengers: {len(mr)}')
print(f'Mr survived:   {mr["survived"].sum()} / {mr["survived"].notna().sum()}')
print(f'Mr age stats:  mean={mr["age"].mean():.1f}, median={mr["age"].median():.1f}')
print(f'Mr parch>0:    {(mr["parch"]>0).sum()}  (has children/parents aboard)')
print(f'Mr sibsp>=1:   {(mr["sibsp"]>=1).sum()}  (has spouse/sibling aboard)')
print()

# If Mr has Surname overlap -> likely has wife aboard 
df['HasWifeAboard'] = df.apply(
    lambda r: 1 if (r['Title']=='Mr' and r['Surname'] in mrs_surnames) else 0, axis=1)
print('HasWifeAboard distribution:')
print(df.groupby('HasWifeAboard')[['survived']].agg(['mean','count']))
print()

# Age-based children proxy: if Mr with parch>0, estimate avg child age
mr_with_kids = mr[mr['parch']>0]
print(f'Mr with parch>0: {len(mr_with_kids)} (likely has children aboard)')
# Proxy: father age - 25 = rough first-child age estimate
# If father is 35 -> child ~10 years old (young)
# If father is 50 -> child ~25 years old (adult)
mr_with_kids = mr_with_kids.copy()
mr_with_kids['EstChildAge'] = mr_with_kids['age'] - 25
print(mr_with_kids[['age','parch','EstChildAge','survived']].describe())
