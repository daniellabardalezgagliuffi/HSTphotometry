import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#python3 HSTplots.py --photfile=HSTwisespitzerphot_2020Apr03.csv


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='make figures')
    parser.add_argument('--photfile', dest='photfile', help='HST photometry file')
    args = parser.parse_args()
    hstphot = pd.read_csv(args.photfile).drop('Unnamed: 0',1)
    date = args.photfile.split('_')[1].split('.csv')[0]
    print(date)
    
    ##############################
    #### FUNCTION DEFINITIONS ####
    ##############################   

    def ch1ch2_Mch2(ch1,ch2,ch1e,ch2e):

        ch12 = ch1 - ch2
        ch12e = np.sqrt(ch1e**2 + ch2e**2)
        a = 12.9611
        da = 0.771287
        b = -0.595238
        db = 1.22560
        c = 0.554636
        dc = 0.613005
        d = -0.0206577
        dd = 0.0967258
        rms = 0.30
        if (ch12 >= 0.9) and (ch12 <= 3.7):
            Mch2 = a + b*ch12 + c*ch12**2 + d*ch12**3
            Mch2e = np.sqrt(da**4 + db**4*ch12**2 + dc**4*ch12**4 + dd**4*ch12**6 + (b + 2*c*ch12 + 3*d*ch12**2)**2*ch12e**2)
        else:
            Mch2, Mch2e = np.nan, np.nan
        return Mch2,Mch2e


    def ch1ch2_Teff(ch1,ch2,ch1e,ch2e):

        if pd.notnull(ch1) and pd.notnull(ch2):
            ch12 = ch1 - ch2
            ch12e = np.sqrt(ch1e**2 + ch2e**2)
            if (ch12 >= 0.9) & (ch12 <= 3.6):
                rms = 81
                a = 1603.55
                b = -658.290
                c = 79.4503
                Teff = a + b * ch12 + c * ch12**2
                Teff_e = np.sqrt(rms**2 + (b + 2 * c * ch12)**2 * ch12e**2)
            return [Teff, Teff_e]
        else:
            return [np.nan, np.nan]


    def ch1ch2_spt(ch1,ch2):

        from scipy.interpolate import interp1d
        spt = np.arange(6,14,0.5)
        ch1ch2 = 0.769877 - 0.229115*spt + 0.0645336*spt**2 - 0.00245289*spt**3
        fspt = interp1d(ch1ch2,spt,kind='linear', fill_value="extrapolate")

        ch12 = ch1 - ch2
        #ch12e = np.sqrt(ch1e**2 + ch2e**2)
        sptnew = fspt(ch12)

        return sptnew

    def redpm(mag,mage,mutot,mutote):

        #mu = total proper motion in arcsec/yr
        mu = mutot/1000.
        mue = mutote/1000.
        H = mag + 5*np.log10(mu) +5
        Herr = np.sqrt(mage**2 + (5/(np.log(10)*mu)**2*mue**2))
        return [H,Herr]

    def get_dist(M,m,Merr,merr):

        #mag = ufloat(m,merr)
        #Mag = ufloat(M,Merr)
        d = 10**((m-M)/5+1)
        derr = np.sqrt((0.2*np.log(10)*10**((m-M)/5+1)*merr)**2 + (0.2*np.log(10)*10**((m-M)/5+1)*Merr)**2)
        return [d,derr]

    def dist_absmag(ch1,ch1e,ch2,ch2e):

        ch12 = ch1 - ch2
        ch12e = np.sqrt(ch1e**2 + ch2e**2)

        if (ch12  >= 0.9) & (ch12 <= 3.7):
            a = 12.9611
            b = -0.595238
            c = 0.554636
            d = -0.0206577
            rms = 0.30
            Mch2 = a + b * ch12 + c * ch12**2 + d * ch12**3
            Mch2e = np.sqrt(rms**2 + (b + 2 * c * ch12 + 3 * d * ch12**2)**2 * ch12e**2)
            mu = ch2 - Mch2
            mue = np.sqrt(ch2e**2 + Mch2e**2)
            dist = 10**(mu / 5. + 1)
            dist_e = 0.2 * np.log(10) * 10**(0.2 * mu + 1) * mue
        else:
            dist, dist_e, Mch2, Mch2e = np.nan,np.nan,np.nan,np.nan

        return [dist, dist_e, Mch2, Mch2e]

    def absmag(appmag, appmage, plx, plxe):

        M = appmag + 5 - 5*np.log10(1000./plx)
        Me = np.sqrt(appmage**2 + (5*plxe/(plx*np.log(10))**2))
        return [M,Me]

    def Mch2_teff(Mch2, Mch2e):

        # ± 15647.4  ± 3231.62 ± 221.694  ± 5.05173
        rms = 73
        Teff = 35476.5 -6198.65*Mch2 + 366.839*Mch2**2  -7.29548*Mch2**3
        Teffe = np.sqrt(rms**2 + (-6198.65 + 366.839*2*Mch2 -7.29548*3*Mch2**2)**2*Mch2e**2)
        return [Teff, Teffe]

    def Mch2_spt(Mch2):

        #± 1.89109 ± 0.652409 ± 0.0728241  ± 0.00262970
        from scipy.interpolate import interp1d
        spt = np.arange(6,14,0.5)
        absMch2 = 16.3304 -1.56047*spt + 0.203183*spt**2 -0.00635074*spt**3
        fspt = interp1d(absMch2,spt,kind='linear', fill_value="extrapolate")
        sptnew = fspt(Mch2)
        return sptnew

    def vtan(pm, epm, dist, edist):
        #pm in arcsec/yr
        #dist in pc
        vtan = 4.74 * pm * dist
        evtan = vtan * np.sqrt((epm/pm)**2 + (edist/dist)**2)
        return [vtan, evtan]

    w0830 = hstphot[hstphot['SOURCE'] == 'WISE0830+2837'].index
    print(w0830)
    
    hstphot['estspt'] = ch1ch2_spt(hstphot['ch1'],hstphot['ch2'])
    hstphot['plx'] = [np.nan,90.2,np.nan,np.nan,np.nan]
    hstphot['plxe'] = [np.nan,13.7,np.nan,np.nan,np.nan]
    hstphot['trigd'] = 1000/hstphot['plx']
    hstphot['etrigd'] = 1000*hstphot['plxe']/hstphot['plx']**2
    hstphot['pmtot_spz'] = np.nan
    hstphot['epmtot_spz'] = np.nan
    hstphot.loc[w0830,'pmtot_spz'] = np.sqrt(233.3**2 + 2040.8**2) #RA and Dec negatives
    hstphot.loc[w0830,'epmtot_spz'] = np.sqrt(48.6**2 + 29.9**2)
    Mch2 = [ch1ch2_Mch2(hstphot.loc[i,'ch1'],hstphot.loc[i,'ch2'],hstphot.loc[i,'ch1e'],hstphot.loc[i,'ch2e']) for i in range(len(hstphot))]
    hstphot['Mch2'] = [Mch2[i][0] for i in range(len(Mch2))]
    hstphot['Mch2e'] = [Mch2[i][1] for i in range(len(Mch2))]
    hstphot.loc[w0830,'Mch2'] = absmag(hstphot.loc[w0830,'ch2'], hstphot.loc[w0830,'ch2e'], hstphot.loc[w0830,'plx'], hstphot.loc[w0830,'plxe'])[0]
    hstphot.loc[w0830,'estspt'] = Mch2_spt(hstphot.loc[w0830,'Mch2'])
    hstphot['vtan'] = 4.74*hstphot['pmtotCW']/1000*hstphot['dist']
    hstphot['evtan'] = np.nan
    hstphot.loc[w0830,'vtan'] = 4.74*hstphot.loc[w0830,'pmtot_spz']/1000*hstphot.loc[w0830,'trigd']
    hstphot.loc[w0830,'evtan'] = hstphot.loc[w0830,'vtan'] * np.sqrt((hstphot.loc[w0830,'epmtot_spz']/hstphot.loc[w0830,'pmtot_spz'])**2 + (hstphot.loc[w0830,'etrigd']/hstphot.loc[w0830,'trigd'])**2)
    print(hstphot)
    
    hstphot.to_csv('HST_allphot_'+date+'.csv')

    
    #############################
    ##### MANUALLY ADD DATA #####
    #############################

    # Marocco et al. 2019 WISE and Spitzer photometry of source W1935-1546
    fede = {'Name':'1935–1546','w1':18.534,'w1er':0.396,'w2':15.852,'w2er':0.079,'ch1':18.892,'ch1er': 0.314,'ch2': 15.647,'ch2er':0.023,
            'ch1–ch2':3.24,'ch1-ch2e':0.31,'pmra':337,'epmra':69,'pmde':-50,'epmde':97}
    fededf = pd.DataFrame(index=np.arange(1),data=fede)
    fededf['pmtot'] = np.sqrt(fededf['pmra']**2 + fededf['pmde']**2)/1000.
    fededf['Hch2'] = fededf['ch2'] + 5*np.log10(fededf['pmtot']) +5
    
    # Y dwarf compilation
    ydwarfs = pd.read_csv('/Users/daniella/Research/Datasets/HST_Ydwarfs.csv').drop(['Index','RA','DEC','pmra','pmraer','pmdec','pmdecer','ch1-ch2','Dist (est)'],1).rename(columns={'ch1-ch2.1':'ch1-ch2','Spt (est)':'estspt'})

    #Schneider et al. 2015 synhtetic HST photometry of Y dwarfs
    sch15ty = pd.read_csv('/Users/daniella/Research/Relations/Schneider2015_privcomm_hst_synphot.txt',delim_whitespace=True, header=1)

    #Schneider et al. 2015 HST photometry
    sch15 = pd.read_csv('/Users/daniella/Research/Datasets/schneider2015_hst_ydwarfs.txt',delim_whitespace=True,comment='#')

    #Colors for Y dwarf compilation, merge with Fede's photometry of 1935
    cols = ydwarfs.columns
    ydwarfs = pd.concat([ydwarfs,fededf],sort=True).reindex(columns=cols).reset_index().drop('index',1)
    ydwarfs = ydwarfs.replace(-100,np.nan)
    ydwarfs = ydwarfs.replace('#NUM!',np.nan)
    ydwarfs = ydwarfs.apply(lambda x: pd.to_numeric(x,errors='ignore'))
    ydwarfs.loc[ydwarfs['Name'] == 'J1828','Name'] = '1828+2650'
    ydwarfs.loc[ydwarfs['Name'] == 'J0855','Name'] = '0855-0714'
    ydwarfs['W1-W2'] = ydwarfs['w1']-ydwarfs['w2']
    ydwarfs['W1-W2e'] = np.sqrt(ydwarfs['w1er']**2+ydwarfs['w2er']**2)
    ydwarfs['F125W-W2'] = ydwarfs['F125W']-ydwarfs['w2']
    ydwarfs['F125W-W2e'] = np.sqrt(ydwarfs['F125We']**2+ydwarfs['w2er']**2)
    ydwarfs['F125W-ch2'] = ydwarfs['F125W']-ydwarfs['ch2']
    ydwarfs['F125W-ch2e'] = np.sqrt(ydwarfs['F125We']**2+ydwarfs['ch2er']**2)
    ydwarfs['ch1-ch2'] = ydwarfs['ch1']-ydwarfs['ch2']
    ydwarfs['ch1-ch2e'] = np.sqrt(ydwarfs['ch1er']**2+ydwarfs['ch2er']**2)
    ydwarfs['F105W-F125W'] = ydwarfs['F105W']-ydwarfs['F125W']
    ydwarfs['F105W-F125We'] = np.sqrt(ydwarfs['F105We']**2+ydwarfs['F125We']**2)

    ydwarfs['F125-W2'] = ydwarfs['F125']-ydwarfs['w2']
    ydwarfs['F125-W2e'] = np.sqrt(ydwarfs['eF125']**2+ydwarfs['w2er']**2)
    ydwarfs['F125-ch2'] = ydwarfs['F125']-ydwarfs['ch2']
    ydwarfs['F125-ch2e'] = np.sqrt(ydwarfs['eF125']**2+ydwarfs['ch2er']**2)
    ydwarfs['F105-F125'] = ydwarfs['F105']-ydwarfs['F125']
    ydwarfs['F105-F125e'] = np.sqrt(ydwarfs['e105']**2+ydwarfs['eF125']**2)
    
    ydwarfs['est_spt'] = ydwarfs['estspt'] - 20
    ydwarfs = ydwarfs.drop('estspt',1)

    ydwarfs['shortname'] = ydwarfs['Name'].map(lambda x: x.split('J')[1][0:4] + x.split('J')[1][9:14] if len(x) > 12 else x)
    ydwarfs.loc[22,'shortname'] = '0806-661B'

    #Colors for synthetic photometry - Schneider et al. 2015
    sch15ty['F105-F125'] = sch15ty['F105'] - sch15ty['F125']
    sch15ty['F105-F125e'] = np.sqrt(sch15ty['e105']**2 + sch15ty['e125']**2)
    sch15ty['F125-W2'] = sch15ty['F125'] - sch15ty['W2']
    sch15ty['F125-W2e'] = np.sqrt(sch15ty['e125']**2 + sch15ty['eW2']**2)
    sch15ty['F125-ch2'] = sch15ty['F125'] - sch15ty['ch2']
    sch15ty['F125-ch2e'] = np.sqrt(sch15ty['e125']**2 + sch15ty['ech2']**2)
    sch15ty = sch15ty.rename(columns={'id':'shortname'})

    #Colors for photometry - Schneider et al. 2015
    sch15 = sch15.replace('...',np.nan)
    sch15 = sch15.apply(pd.to_numeric,errors='ignore')
    sch15['shortname'] =  sch15['AllWISEName'].map(lambda x: x.split('WISEAJ')[1][0:4]) + sch15['AllWISEName'].map(lambda x: x.split('WISEAJ')[1][9:14])
    sch15['F105W-F125W'] = sch15['F105W'] - sch15['F125W']
    sch15['F105W-F125We'] = np.sqrt(sch15['F105We']**2 + sch15['F125We']**2)
    sch15['F125W-ch2'] = sch15['F125W'] - sch15['ch2']
    sch15['F125W-ch2e'] = np.sqrt(sch15['F125We']**2 + sch15['ch2e']**2)

    sch2015 = sch15.merge(sch15ty,on='shortname')
    sch2015 = sch2015.drop(['ch2_y','ech2'],1)
    sch2015 = sch2015.rename(columns={'ch2_x':'ch2'})
    sch2015['F125W-W2'] = sch2015['F125W'] - sch2015['W2']
    sch2015['F125W-W2e'] = np.sqrt(sch2015['F125We']**2 + sch2015['eW2']**2)
    
    ysamp = pd.merge(ydwarfs,sch2015,on='shortname',how='left')
    ysamp = ysamp.drop(['ch1_y','ch2_y','ch1e','ch2e','ch1-ch2_y','F105_y','e105_y','F105W_y','F105We_y','F125W_y','F125We_y','F125_y','e125','F125W-W2_y','F125W-W2e_y',
                        'F125W-ch2_y','F125W-ch2e_y','ch1-ch2e_y','F105W-F125W_y','F105W-F125We_y','F125-W2_y','F125-W2e_y','F125-ch2_y','F125-ch2e_y','F105-F125_y',
                        'F105-F125e_y','W2','eW2','AllWISEName'],axis=1)
    ysamp = ysamp.rename(columns={'ch1_x':'ch1', 'ch1er':'ch1e', 'ch2_x':'ch2', 'ch2er':'ch2e', 'ch1-ch2_x':'ch1-ch2', 'F105_x':'F105', 'e105_x':'F105e', 'F105W_x':'F105W',
                                  'F105We_x':'F105We', 'F125W_x':'F125W', 'F125We_x':'F125We', 'F125_x':'F125', 'eF125':'F125e', 'F105 - ch2':'F105-ch2',
                                  'F125 - ch2':'F125-ch2', 'F160 - ch2':'F160-ch2', 'F125W-W2_x':'F125W-W2', 'F125W-W2e_x':'F125W-W2e', 'F125W-ch2_x':'F125W-ch2', 
                                  'F125W-ch2e_x':'F125W-ch2e', 'ch1-ch2e_x':'ch1-ch2e', 'F105W-F125W_x':'F105W-F125W', 'F105W-F125We_x':'F105W-F125We', 'F125-W2_x':'F125-W2',
                                  'F125-W2e_x':'F125-W2e', 'F125-ch2_x':'F125-ch2', 'F125-ch2e_x':'F125-ch2e', 'F105-F125_x':'F105-F125', 'F105-F125e_x':'F105-F125e','est_spt':'estspt'})
    ysamp.loc[ysamp['shortname'] == '0855−0714','estspt'] = 14
    
    ysamp.to_csv('ydwarfphot'+date+'.csv')
    
    t = ysamp[ysamp['spt'] < 10].index
    y = ysamp[ysamp['spt'] >= 10].index
    estt = ysamp[(ysamp['estspt'] < 10) & (ysamp['spt'].isnull())].index
    esty = ysamp[(ysamp['estspt'] >= 10)  & (ysamp['spt'].isnull())].index
    
    #############################
    ########## FIGURES ##########
    #############################

    
    print('Fig 1:')
    
    hstphot.loc[hstphot['F125W-ch2e'].isnull(), 'F125W-ch2e'] = 0.5
    
    fig = plt.figure(figsize=(10,8))

    plt.errorbar(ysamp['spt'],ysamp['F125W-ch2'],yerr=ysamp['F125W-ch2e'],linestyle='None',zorder=1,ecolor='k',elinewidth=1)
    plt.scatter(ysamp.loc[t,'spt'],ysamp.loc[t,'F125W-ch2'],marker='o',s=100,c=sns.xkcd_rgb['cornflower blue'],edgecolor='k')
    plt.scatter(ysamp.loc[y,'spt'],ysamp.loc[y,'F125W-ch2'],marker='o',s=100,c=sns.xkcd_rgb['marigold'],edgecolor='k')

    plt.errorbar(hstphot['estspt'],hstphot['F125W-ch2'], yerr=hstphot['F125W-ch2e'],lolims=[0,1,0,0,0],linestyle='None',zorder=1,ecolor='k',elinewidth=1)
    plt.scatter(hstphot['estspt'],hstphot['F125W-ch2'],marker='o',s=200,c=sns.xkcd_rgb['pale red'],edgecolor='k')
    
    plt.errorbar(ysamp.loc[ysamp['spt'].isnull(),'estspt'],ysamp.loc[ysamp['spt'].isnull(),'F125W-ch2'],linestyle='None',zorder=1,ecolor='k',elinewidth=1)
    plt.scatter(ysamp.loc[estt,'estspt'],ysamp.loc[estt,'F125W-ch2'], marker='o',s=100,c=sns.xkcd_rgb['cornflower blue'],edgecolor='k')
    plt.scatter(ysamp.loc[esty,'estspt'],ysamp.loc[esty,'F125W-ch2'], marker='o',s=100,c=sns.xkcd_rgb['marigold'],edgecolor='k')

    plt.annotate(str(hstphot.loc[0,'NAME'][3:]),xy=(hstphot.loc[0,'spt']+0.22,hstphot.loc[0,'F125W-ch2']-0.15),fontsize=12)
    plt.annotate(str(hstphot.loc[1,'SOURCE'][4:]),xy=(hstphot.loc[1,'spt']-0.45,hstphot.loc[1,'F125W-ch2']-0.1),fontsize=12)
    plt.annotate(str(hstphot.loc[2,'NAME'][3:]),xy=(hstphot.loc[2,'spt']+0.17,hstphot.loc[2,'F125W-ch2']-0.1),fontsize=12)
    plt.annotate(str(hstphot.loc[3,'NAME'][3:]),xy=(hstphot.loc[3,'spt']+0.2,hstphot.loc[3,'F125W-ch2']-0.1),fontsize=12)
    plt.annotate(str(hstphot.loc[4,'NAME'][3:]),xy=(hstphot.loc[4,'spt']+0.08,hstphot.loc[4,'F125W-ch2']-0.15),fontsize=12)
    plt.annotate(str(ysamp.loc[26,'shortname']),xy=(ysamp.loc[26,'estspt']+0.1,ysamp.loc[26,'F125W-ch2']-0.1),fontsize=12)
    plt.annotate(str(ysamp.loc[27,'shortname']),xy=(ysamp.loc[27,'estspt']-0.95,ysamp.loc[27,'F125W-ch2']-0.1),fontsize=12)

    plt.xticks(np.array([8,9,10,11,14]),['T8','T9','Y0','Y1','$\geq$Y4'],fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('F125W$-$ch2 (mag)',fontsize=16)
    plt.xlabel('Estimated Spectral Type',fontsize=16)
    plt.legend(['$HST$ T dwarfs','$HST$ Y dwarfs','BYW'],fontsize=16,loc=4)
    plt.savefig('test_spt_F125W_ch2.jpg',dpi=1000)

    print('Finished plot 1!')
    
    #############################

    print('Fig 2:')

    fig = plt.figure(figsize=(10,8))

    hstphot.loc[hstphot['F105W-F125We'].isnull(), 'F105W-F125We'] = 0.5
    
    plt.errorbar(hstphot['F125W-ch2'],hstphot['F105W-F125W'],xerr=hstphot['F125W-ch2e'],yerr=hstphot['F105W-F125We'],
                 linestyle='None',fmt='o',color='None', ecolor='k', zorder=1,elinewidth=0.8, uplims=[0,1,0,0,0],xlolims=[0,1,0,0,0],lolims=[0,1,0,0,0])
    plt.scatter(hstphot['F125W-ch2'],hstphot['F105W-F125W'],marker='o',linestyle='None',color=sns.xkcd_rgb['pale red'],s=200,edgecolor='k')

    plt.errorbar(ysamp.loc[t,'F125W-ch2'],ysamp.loc[t,'F105W-F125W'],xerr=ysamp.loc[t,'F125W-ch2e'], yerr=ysamp.loc[t,'F105W-F125We'],linestyle='None',fmt='o',color='None', ecolor='k',zorder=1,elinewidth=1)
    plt.scatter(ysamp.loc[t,'F125W-ch2'],ysamp.loc[t,'F105W-F125W'],linestyle='None',color=sns.xkcd_rgb['cornflower blue'],s=100,edgecolor='k')
    plt.errorbar(ysamp.loc[y,'F125W-ch2'],ysamp.loc[y,'F105W-F125W'],xerr=ysamp.loc[y,'F125W-ch2e'], yerr=ysamp.loc[y,'F105W-F125We'],linestyle='None',fmt='o',color='None', ecolor='k',zorder=1,elinewidth=1)
    plt.scatter(ysamp.loc[y,'F125W-ch2'],ysamp.loc[y,'F105W-F125W'],linestyle='None',color=sns.xkcd_rgb['marigold'],s=100,edgecolor='k')

    plt.errorbar(ysamp.loc[ysamp['spt'].isnull(),'F125W-ch2'],ysamp.loc[ysamp['spt'].isnull(),'F105W-F125W'],linestyle='None',zorder=1,ecolor='k',elinewidth=1)
    plt.scatter(ysamp.loc[estt,'F125W-ch2'],ysamp.loc[estt,'F105W-F125W'], marker='o',s=100,c=sns.xkcd_rgb['cornflower blue'],edgecolor='k')
    plt.scatter(ysamp.loc[esty,'F125W-ch2'],ysamp.loc[esty,'F105W-F125W'], marker='o',s=100,c=sns.xkcd_rgb['marigold'],edgecolor='k')

#    plt.errorbar(ydwarfs.loc[t,'F125W-ch2'],ydwarfs.loc[t,'F105W-F125W'],xerr=ydwarfs.loc[t,'F125W-ch2e'], yerr=ydwarfs.loc[t,'F105W-F125We'],linestyle='None',fmt='o',color='None', ecolor='k',zorder=1,elinewidth=1)
#    plt.scatter(ydwarfs.loc[t,'F125W-ch2'],ydwarfs.loc[t,'F105W-F125W'],linestyle='None',color=sns.xkcd_rgb['cornflower blue'],s=100,edgecolor='k')
#    plt.errorbar(ydwarfs.loc[y,'F125W-ch2'],ydwarfs.loc[y,'F105W-F125W'],xerr=ydwarfs.loc[y,'F125W-ch2e'], yerr=ydwarfs.loc[y,'F105W-F125We'],linestyle='None',fmt='o',color='None', ecolor='k',zorder=1,elinewidth=1)
#    plt.scatter(ydwarfs.loc[y,'F125W-ch2'],ydwarfs.loc[y,'F105W-F125W'],linestyle='None',color=sns.xkcd_rgb['marigold'],s=100,edgecolor='k')

#plt.errorbar(ydwarfs.loc[(ydwarfs['F125W'].isnull()) & (ydwarfs['estspt'] < 40), 'F125-ch2'],ydwarfs.loc[(ydwarfs['F125W'].isnull()) & (ydwarfs['estspt'] < 40), 'F105-F125'],xerr=ydwarfs.loc[ydwarfs['F125W'].isnull() & (ydwarfs['estspt'] < 40),'F125-ch2e'], yerr=ydwarfs.loc[(ydwarfs['F125W'].isnull()) & (ydwarfs['estspt'] < 40),'F105-F125e'],linestyle='None',fmt='o',color='None', ecolor='k',zorder=1,elinewidth=1)
#    plt.scatter(ydwarfs.loc[(ydwarfs['F125W'].isnull()) & (ydwarfs['estspt'] < 40), 'F125-ch2'],ydwarfs.loc[(ydwarfs['F125W'].isnull()) & (ydwarfs['estspt'] < 40), 'F105-F125'],linestyle='None',color=sns.xkcd_rgb['cornflower blue'],s=100,label='$HST$ T dwarfs',edgecolor='k')

 #   plt.errorbar(ydwarfs.loc[(ydwarfs['F125W'].isnull()) & (ydwarfs['estspt'] >= 40), 'F125-ch2'],ydwarfs.loc[(ydwarfs['F125W'].isnull()) & (ydwarfs['estspt']  >= 40), 'F105-F125'],xerr=ydwarfs.loc[ydwarfs['F125W'].isnull() & (ydwarfs['estspt']  >= 40),'F125-ch2e'], yerr=ydwarfs.loc[(ydwarfs['F125W'].isnull()) & (ydwarfs['estspt']  >= 40),'F105-F125e'],linestyle='None',fmt='o',color='None', ecolor='k',zorder=1,elinewidth=1)
 #   plt.scatter(ydwarfs.loc[(ydwarfs['F125W'].isnull()) & (ydwarfs['estspt']  >= 40), 'F125-ch2'],ydwarfs.loc[(ydwarfs['F125W'].isnull()) & (ydwarfs['estspt']  >= 40), 'F105-F125'],linestyle='None',color=sns.xkcd_rgb['marigold'],s=100,label='$HST$ Y dwarfs',edgecolor='k')

    plt.annotate(hstphot.loc[0,'SOURCE'][4:13],xy=(hstphot.loc[0,'F125W-ch2']+0.2,hstphot.loc[0,'F105W-F125W']-0.02),fontsize=12)
    plt.annotate(hstphot.loc[1,'SOURCE'][4:13],xy=(hstphot.loc[1,'F125W-ch2']+0.2,hstphot.loc[1,'F105W-F125W']-0.02),fontsize=12)
    plt.annotate(hstphot.loc[2,'SOURCE'][4:13],xy=(hstphot.loc[2,'F125W-ch2']-1.6,hstphot.loc[2,'F105W-F125W']-0.02),fontsize=12)
    plt.annotate(hstphot.loc[3,'SOURCE'][4:13],xy=(hstphot.loc[3,'F125W-ch2']-1.7,hstphot.loc[3,'F105W-F125W']-0.02),fontsize=12)
    plt.annotate(hstphot.loc[4,'SOURCE'][4:13],xy=(hstphot.loc[4,'F125W-ch2']+0.25,hstphot.loc[4,'F105W-F125W']-0.02),fontsize=12)

    plt.annotate(ysamp.loc[26,'shortname'],xy=(ysamp.loc[26,'F125W-ch2']+0.2,ysamp.loc[26,'F105W-F125W']-0.02),fontsize=12)
    plt.annotate(ysamp.loc[27,'shortname'],xy=(ysamp.loc[27,'F125W-ch2']-1.6,ysamp.loc[27,'F105W-F125W']+0.01),fontsize=12)
    plt.annotate(ysamp.loc[2,'shortname'],xy=(ysamp.loc[2,'F125W-ch2']-1.6,ysamp.loc[2,'F105W-F125W']+0.01),fontsize=12)

    plt.ylim(-0.4,1.2)
    plt.xlim(2,14)
    plt.xlabel('F125W$-$ch2 (mag)',fontsize=16)
    plt.ylabel('F105W$-$F125W (mag)',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(['BYW','$HST$ T dwarfs','$HST$ Y dwarfs'],fontsize=16)
    plt.savefig('test_F125W_ch2_F105W_F125W.jpg',dpi=1000)

    print('Finished plot 2')
    
    #############################
    
    print('Fig 3:')
    
    hstphot.loc[hstphot['F125W-ch2e'].isnull(), 'F125W-ch2e'] = 0.5
    
    fig = plt.figure(figsize=(10,8))

    plt.errorbar(hstphot['ch1-ch2'],hstphot['F125W-ch2'],yerr=hstphot['F125W-ch2e'],xerr=hstphot['ch1-ch2e'],
                 linestyle='None',fmt='o',color='None', ecolor='k', zorder=1, elinewidth=0.8, lolims=[0,1,0,0,0])
    plt.scatter(hstphot['ch1-ch2'],hstphot['F125W-ch2'],marker='o',linestyle='None',color=sns.xkcd_rgb['pale red'],s=200,edgecolor='k')

    plt.errorbar(ysamp.loc[t,'ch1-ch2'],ysamp.loc[t,'F125W-ch2'],yerr=ysamp.loc[t,'F125W-ch2e'], xerr=ysamp.loc[t,'ch1-ch2e'],
                 linestyle='None',fmt='o',color='None', ecolor='k',zorder=1,elinewidth=1)
    plt.scatter(ysamp.loc[t,'ch1-ch2'],ysamp.loc[t,'F125W-ch2'],linestyle='None',color=sns.xkcd_rgb['cornflower blue'],s=100,edgecolor='k')
    plt.errorbar(ysamp.loc[y,'ch1-ch2'],ysamp.loc[y,'F125W-ch2'],yerr=ysamp.loc[y,'F125W-ch2e'], xerr=ysamp.loc[y,'ch1-ch2e'],
                 linestyle='None',fmt='o',color='None', ecolor='k',zorder=1,elinewidth=1)
    plt.scatter(ysamp.loc[y,'ch1-ch2'],ysamp.loc[y,'F125W-ch2'],linestyle='None',color=sns.xkcd_rgb['marigold'],s=100,edgecolor='k')

    plt.errorbar(ysamp.loc[ysamp['spt'].isnull(),'ch1-ch2'],ysamp.loc[ysamp['spt'].isnull(),'F125W-ch2'],
                 xerr=ysamp.loc[ysamp['spt'].isnull(),'ch1-ch2e'],yerr=ysamp.loc[ysamp['spt'].isnull(),'F125W-ch2e'],
                 linestyle='None',zorder=1,ecolor='k',elinewidth=1)
    plt.scatter(ysamp.loc[estt,'ch1-ch2'], ysamp.loc[estt,'F125W-ch2'],marker='o',s=100,c=sns.xkcd_rgb['cornflower blue'],edgecolor='k')
    plt.scatter(ysamp.loc[esty,'ch1-ch2'], ysamp.loc[esty,'F125W-ch2'],marker='o',s=100,c=sns.xkcd_rgb['marigold'],edgecolor='k')

    plt.annotate(hstphot.loc[0,'SOURCE'][4:13],xy=(hstphot.loc[0,'ch1-ch2']-0.1, hstphot.loc[0,'F125W-ch2']+0.25),fontsize=12)
    plt.annotate(hstphot.loc[1,'SOURCE'][4:13],xy=(hstphot.loc[1,'ch1-ch2']+0.03, hstphot.loc[1,'F125W-ch2']+0.2),fontsize=12)
    plt.annotate(hstphot.loc[2,'SOURCE'][4:13],xy=(hstphot.loc[2,'ch1-ch2']+0.075, hstphot.loc[2,'F125W-ch2']-0.08),fontsize=12)
    plt.annotate(hstphot.loc[3,'SOURCE'][4:13],xy=(hstphot.loc[3,'ch1-ch2']+0.1, hstphot.loc[3,'F125W-ch2']-0.08),fontsize=12)
    plt.annotate(hstphot.loc[4,'SOURCE'][4:13],xy=(hstphot.loc[4,'ch1-ch2']+0.1, hstphot.loc[4,'F125W-ch2']-0.08),fontsize=12)

    plt.annotate(ysamp.loc[26,'shortname'],xy=(ysamp.loc[26,'ch1-ch2']+0.06, ysamp.loc[26,'F125W-ch2']-0.1),fontsize=12)
    plt.annotate(ysamp.loc[27,'shortname'],xy=(ysamp.loc[27,'ch1-ch2']-0.32, ysamp.loc[27,'F125W-ch2']-0.1),fontsize=12)


    plt.ylabel('F125W$-$ch2 (mag)',fontsize=16)
    plt.xlabel('ch1$-$ch2 (mag)',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(['BYW','$HST$ T dwarfs','$HST$ Y dwarfs'],fontsize=16)
    plt.savefig('ch1ch2_F125W_ch2.jpg',dpi=1000)

    print('Finished plot 3')

    
    #############################
 
