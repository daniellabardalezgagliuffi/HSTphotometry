import numpy as np
import matplotlib.pyplot as plt
import aplpy
import pyfits

ra1 = 127.5497878; de1 = 28.6211747 # CatWISE Position
ra2 = 127.5493542; de2 = 28.6164217 # Spitzer Position
ra3 = 127.5493833; de3 = 28.6166464 # HST position

path = '/Users/daniella/Research/PythonProjects/BYW_HSTphotometry/images/'

plt.ion()
fig = plt.figure(figsize=(14,9))
#plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95,wspace=0.05,hspace=0.05)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.1,hspace=0.1)
#pylab.rcParams['font.family'] = 'serif'

data1 = pyfits.getdata(path+'unwise-1268p287-w2-img-m.fits')
med1 = np.nanmedian(data1)
mad1 = np.nanmedian(abs(data1 - np.nanmedian(data1)))
min1 = med1 - 2.0*mad1
max1 = med1 + 10.0*mad1

data2 = pyfits.getdata(path+'unwise-1268p287-w2-img-m_post.fits')
med2 = np.nanmedian(data2)
mad2 = np.nanmedian(abs(data2 - np.nanmedian(data2))) 
min2 = med2 - 2.0*mad2
max2 = med2 + 10.0*mad2 

data3 = pyfits.getdata(path+'idt702010_drz.fits')
med3 = np.nanmedian(data3)
mad3 = np.nanmedian(abs(data3 - np.nanmedian(data3))) 
min3 = med3 - 2.0*mad3
max3 = med3 + 10.0*mad3 

data4 = pyfits.getdata(path+'idt702020_drz.fits')
med4 = np.nanmedian(data4)
mad4 = np.nanmedian(abs(data4 - np.nanmedian(data4)))
min4 = med4 - 2.0*mad4
max4 = med4 + 10.0*mad4

data5 = pyfits.getdata(path+'SPITZER_I1_65975296_0000_1_E12688204_maic.fits')
med5 = np.nanmedian(data5)
mad5 = np.nanmedian(abs(data5 - np.nanmedian(data5)))
min5 = med5 - 2.0*mad5
max5 = med5 + 10.0*mad5

data6 = pyfits.getdata(path+'SPITZER_I2_65975296_0000_1_E12688162_maic.fits')
med6 = np.nanmedian(data6)
mad6 = np.nanmedian(abs(data6 - np.nanmedian(data6)))
min6 = med6 - 2.0*mad6
max6 = med6 + 10.0*mad6

im20 = aplpy.FITSFigure(path+'unwise-1268p287-w2-img-m.fits',figure=fig,subplot=(2,3,1),north=True)
im20.tick_labels.hide()
im20.axis_labels.hide()
im20.ticks.set_color('k')
im20.ticks.set_minor_frequency(1)
im20.ticks.set_xspacing(0.5/60.0)
im20.ticks.set_xspacing(0.5/60.0)
im20.ticks.set_yspacing(0.5/60.0)
im20.tick_labels.hide_x()
im20.tick_labels.hide_y()
im20.show_colorscale(cmap='gist_yarg',aspect='equal',vmax=max1,vmin=min1)
im20.add_scalebar((0.1/60.0),color='k')
im20.scalebar.set_corner('bottom right')
im20.scalebar.set_linewidth(2)
im20.scalebar.set_label("  0.1'")
#im20.add_label(0.25,0.9,'unWISE epoch 1',relative=True,size='large',color='k')
im20.recenter(ra1,de1,width=(1.0/60.0),height=(1.0/60.0))
im20.show_circles(ra1,de1,edgecolor='g',facecolor='none',radius=0.0010,linewidth=1.5)
im20.show_circles(ra2,de2,edgecolor='r',facecolor='none',radius=0.0010,linewidth=1.5)
im20.show_circles(ra3,de3,edgecolor='b',facecolor='none',radius=0.0010,linewidth=1.5)
plt.title('unWISE W2 2010-07-21', fontsize=16)

im20 = aplpy.FITSFigure(path+'unwise-1268p287-w2-img-m_post.fits',figure=fig,subplot=(2,3,4),north=True)
im20.tick_labels.hide()
im20.axis_labels.hide()
im20.ticks.set_color('k')
im20.ticks.set_minor_frequency(1)
im20.ticks.set_xspacing(0.5/60.0)
im20.ticks.set_xspacing(0.5/60.0)
im20.ticks.set_yspacing(0.5/60.0)
im20.tick_labels.hide_x()
im20.tick_labels.hide_y()
im20.show_colorscale(cmap='gist_yarg',aspect='equal',vmax=max2,vmin=min2)
#im20.add_label(0.25,0.9,'unWISE epoch 2',relative=True,size='large',color='k')
im20.recenter(ra1,de1,width=(1.0/60.0),height=(1.0/60.0))
im20.show_circles(ra1,de1,edgecolor='g',facecolor='none',radius=0.0010,linewidth=1.5)
im20.show_circles(ra2,de2,edgecolor='r',facecolor='none',radius=0.0010,linewidth=1.5)
im20.show_circles(ra3,de3,edgecolor='b',facecolor='none',radius=0.0010,linewidth=1.5)
plt.title('unWISE W2 2015-01-20', fontsize=16)

im20 = aplpy.FITSFigure(path+'idt702010_drz.fits',figure=fig,subplot=(2,3,2),north=True)
im20.tick_labels.hide()
im20.axis_labels.hide()
im20.ticks.set_color('k')
im20.ticks.set_minor_frequency(1)
im20.ticks.set_xspacing(0.5/60.0)
im20.ticks.set_xspacing(0.5/60.0)
im20.ticks.set_yspacing(0.5/60.0)
im20.tick_labels.hide_x()
im20.tick_labels.hide_y()
#im20.show_grayscale(invert=True,stretch='arcsinh')#vmin[i],vmax[i],
im20.show_colorscale(cmap='gist_yarg',aspect='equal',vmax=max3,vmin=min3)
#im20.add_label(0.25,0.9,'HST F105W',relative=True,size='large',color='k')
im20.recenter(ra1,de1,width=(1.0/60.0),height=(1.0/60.0))
im20.show_circles(ra1,de1,edgecolor='g',facecolor='none',radius=0.0010,linewidth=1.7)
im20.show_circles(ra2,de2,edgecolor='r',facecolor='none',radius=0.0010,linewidth=1.7)
im20.show_circles(ra3,de3,edgecolor='b',facecolor='none',radius=0.0010,linewidth=1.7)
plt.title('HST F105W 2018-09-29', fontsize=16)

im20 = aplpy.FITSFigure(path+'idt702020_drz.fits',figure=fig,subplot=(2,3,5),north=True)
im20.tick_labels.hide()
im20.axis_labels.hide()
im20.ticks.set_color('k')
im20.ticks.set_minor_frequency(1)
im20.ticks.set_xspacing(0.5/60.0)
im20.ticks.set_xspacing(0.5/60.0)
im20.ticks.set_yspacing(0.5/60.0)
im20.tick_labels.hide_x()
im20.tick_labels.hide_y()
im20.show_colorscale(cmap='gist_yarg',aspect='equal',vmax=max4,vmin=min4)
#im20.add_label(0.25,0.9,'HST F125W',relative=True,size='large',color='k')
im20.recenter(ra1,de1,width=(1.0/60.0),height=(1.0/60.0))
im20.show_circles(ra1,de1,edgecolor='g',facecolor='none',radius=0.0010,linewidth=1.5)
im20.show_circles(ra2,de2,edgecolor='r',facecolor='none',radius=0.0010,linewidth=1.5)
im20.show_circles(ra3,de3,edgecolor='b',facecolor='none',radius=0.0010,linewidth=1.5)
plt.title('HST F125W 2018-09-29', fontsize=16)

im20 = aplpy.FITSFigure(path+'SPITZER_I1_65975296_0000_1_E12688204_maic.fits',figure=fig,subplot=(2,3,3),north=True)
im20.tick_labels.hide()
im20.axis_labels.hide()
im20.ticks.set_color('k')
im20.ticks.set_minor_frequency(1)
im20.ticks.set_xspacing(0.5/60.0)
im20.ticks.set_xspacing(0.5/60.0)
im20.ticks.set_yspacing(0.5/60.0)
im20.tick_labels.hide_x()
im20.tick_labels.hide_y()
im20.show_colorscale(cmap='gist_yarg',aspect='equal',vmin=0.09,vmax=0.2)
#im20.add_label(0.25,0.9,'Spitzer ch1',relative=True,size='large',color='k')
im20.recenter(ra1,de1,width=(1.0/60.0),height=(1.0/60.0))
im20.show_circles(ra1,de1,edgecolor='g',facecolor='none',radius=0.0010,linewidth=1.5)
im20.show_circles(ra2,de2,edgecolor='r',facecolor='none',radius=0.0010,linewidth=1.5)
im20.show_circles(ra3,de3,edgecolor='b',facecolor='none',radius=0.0010,linewidth=1.5)
plt.title('Spitzer ch1 2019-02-21', fontsize=16)

im20 = aplpy.FITSFigure(path+'SPITZER_I2_65975296_0000_1_E12688162_maic.fits',figure=fig,subplot=(2,3,6),north=True)
im20.tick_labels.hide()
im20.axis_labels.hide()
im20.ticks.set_color('k')
im20.ticks.set_minor_frequency(1)
im20.ticks.set_xspacing(0.5/60.0)
im20.ticks.set_xspacing(0.5/60.0)
im20.ticks.set_yspacing(0.5/60.0)
im20.tick_labels.hide_x()
im20.tick_labels.hide_y()
im20.show_colorscale(cmap='gist_yarg',aspect='equal',vmin=0.26,vmax=0.4)
#im20.add_label(0.25,0.9,'Spitzer ch2',relative=True,size='large',color='k')
im20.recenter(ra1,de1,width=(1.0/60.0),height=(1.0/60.0))
im20.show_circles(ra1,de1,edgecolor='g',facecolor='none',radius=0.0010,linewidth=1.5)
im20.show_circles(ra2,de2,edgecolor='r',facecolor='none',radius=0.0010,linewidth=1.5)
im20.show_circles(ra3,de3,edgecolor='b',facecolor='none',radius=0.0010,linewidth=1.5)
plt.title('Spitzer ch2 2019-02-21', fontsize=16)

plt.savefig('W0830_finder1.pdf')
#raw_input('Enter to exit...')
