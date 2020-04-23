import numpy as np
import pandas as pd
import datetime as dt

w0014 = pd.read_csv('data/W0014+7915/WISE0014+7951_summary.csv')
w0830p = pd.read_csv('data/W0830+2837/WISE0830+2837_summary.csv')
w0830m = pd.read_csv('data/W0830-6323/WISE0830-6323_summary.csv')
w1516 = pd.read_csv('data/W1516+7217/WISE1516+7217_summary.csv')
w1525 = pd.read_csv('data/W1525+6053/WISE1525+6053_summary.csv')

today = dt.datetime.today().strftime("%Y%b%d")

df = pd.concat([w0014, w0830p, w0830m, w1516, w1525],axis=0)
df.to_csv('HSTphotometry_'+str(today) + '.csv', index=False)

print('done!')
print('file saved at HSTphotometry_'+str(today) + '.csv')
print('HST photometry only')

newdf = pd.DataFrame(index=np.arange(5),columns=['SOURCE','F105W','F105We','F125W','F125We'])

newdf['SOURCE'] = df['TARGNAME'].unique()
newdf['F105W'] = df.loc[df['FILTER'] == 'F105W','VEGAMAG'].values
newdf['F105We'] = df.loc[df['FILTER'] == 'F105W','VEGAMAG_UNC'].values
newdf['F125W'] = df.loc[df['FILTER'] == 'F125W','VEGAMAG'].values
newdf['F125We'] = df.loc[df['FILTER'] == 'F125W','VEGAMAG_UNC'].values
newdf.to_csv('HSTphotometry_only_'+str(today) + '.csv')

print('file saved at HSTphotometry_only_'+str(today) + '.csv')
print('reading WISE photometry')

#WISE photometry of the sample
wise = pd.read_csv('~/Research/PythonProjects/BYW_HSTphotometry/wisetable.csv').drop('Unnamed: 0',1)

newphot = newdf.merge(wise,on='SOURCE')
newphot['F105W-F125W'] = newphot['F105W'] - newphot['F125W']
newphot['F105W-F125We'] = np.sqrt(newphot['F105We']**2 + newphot['F125We']**2)
newphot['F125W-W2'] = newphot['F125W'] - newphot['W2']
newphot['F125W-W2e'] = np.sqrt(newphot['F125We']**2 + newphot['W2e']**2)
newphot['F125W-W2cat'] = newphot['F125W'] - newphot['W2cat']
newphot['F125W-W2ecat'] = np.sqrt(newphot['F125We']**2 + newphot['W2ecat']**2)

print('reading Spitzer photometry')

#Spitzer photometry of the sample
spz = pd.read_csv('/Users/daniella/Research/Datasets/Kirk_unpub_results_input_photometry_with_dist_type_estimates_BYW_all.txt',delim_whitespace=True,comment='#')
subspz = spz[(spz['NAME'] == 'BYW0014+7951') | (spz['NAME'] == 'BYW0830+2836') | (spz['NAME'] == 'BYW1525+6054') |
          (spz['NAME'] == 'BYW0830-6323') | (spz['NAME'] == 'BYW1516+7217')]
subspz = subspz.apply(lambda x: pd.to_numeric(x,errors='ignore'))
subspz = subspz[['NAME','ch1_PRF(mag)','ech1_PRF(mag)','ch2_PRF(mag)','ech2_PRF(mag)','ch1-ch2_PRF','ech1-ch2_PRF','spt','dist']].reset_index().drop('index',1)
subspz['SOURCE'] = ['WISE0014+7951', 'WISE0830+2837', 'WISE0830-6323', 'WISE1516+7217', 'WISE1525+6053']
subspz = subspz.rename(columns={'ch1_PRF(mag)':'ch1','ech1_PRF(mag)':'ch1e','ch2_PRF(mag)':'ch2','ech2_PRF(mag)':'ch2e','ch1-ch2_PRF':'ch1-ch2','ech1-ch2_PRF':'ch1-ch2e'})

print('merge HST, WISE, Spitzer')

hstphot = pd.merge(newphot, subspz, on='SOURCE')
hstphot['F125W-ch2'] = hstphot['F125W'] - hstphot['ch2']
hstphot['F125W-ch2e'] = np.sqrt(hstphot['F125We']**2 + hstphot['ch2e']**2)

hstphot.to_csv('HSTwisespitzerphot_'+str(today)+'.csv')
print('file saved at HSTwisespitzerphot_'+str(today)+'.csv')

hst = hstphot[['SOURCE', 'F105W', 'F105We', 'F125W', 'F125We', 'F105W-F125W', 'F105W-F125We', 'ch1', 'ch1e', 'ch2', 'ch2e', 'ch1-ch2', 'ch1-ch2e']]
hst.transpose().to_latex('w0830_tab3.tex')
