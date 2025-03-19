import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore") # Silence warnings
import pandas as pd
import sys
import scipy as sp
import scipy.ndimage
from itertools import groupby,islice
import Ngl
from waveFilter import kf_filter
from kf_filter_plus_PCF_projection import kf_filter_plus_PCF_projection
import os.path

def get_itcz_coords_given_date(fname,date,lon_shift):
    
    REarth = 6358.0
    
    #start a dataset that we will save as a dataframe for later analysis
    data = {
        'time': [],
        'lat': [],
        'lon': [],
        'id': []
    }

    def gcdist(lon1,lons,lat1,lats):
        lon1 = np.radians(lon1)
        lons = np.radians(lons)
        lat1 = np.radians(lat1)
        lats = np.radians(lats)
        # great circle distance. 
        arg = np.sin(lat1)*np.sin(lats)+np.cos(lat1)*np.cos(lats)*np.cos(lon1-lons)

        # Ellipsoid [CLARKE 1866]  Semi-Major Axis (Equatorial Radius)
        a = 6378.2064
        return np.arccos(arg)* a
    
    def read_vars_for_itcz_id(fname,keep_vars):

        ds = xr.open_dataset(fname)

        #convert lat and lon to radians
        ds['lat'] = xr.ufuncs.deg2rad(ds.lat)
        ds['lon'] = xr.ufuncs.deg2rad(ds.lon)
    #     ds['fac'] = xr.ufuncs.tan(ds.lat)
    #    dv = (1.0/REarth)*U.differentiate(coord='lon')+(1.0/REarth)*V.differentiate(coord='lat')-V*ds.fac/REarth
    #     ds['dv'] = sp.ndimage.filters.gaussian_filter(dv, sigma, mode='constant')

        #compute wet bulb potential temp
        if ('thw' in keep_vars):
            tk = ds.tk850
            qv = ds.qv850
            rh = ds.rh850
            print('here')
            tc = tk-273.15
            tw = tc*np.arctan(0.151977*(rh+8.313659)**0.5)+np.arctan(tc+rh)-np.arctan(rh-1.676331)+0.00391838*(rh)**(1.5)*np.arctan(0.023101*rh)-4.686035
            ds['thw'] = (tw+273.15)*(1000.0/850.0)**(287.15/1004.0)
        #keep only the variables that we need: tw, u, and v
        ds = ds.drop([v for v in ds.variables if v not in keep_vars])

        return ds  

    def line_joining_algorithm(pot_itcz_coords):
        REarth=6378.0
        #we found a line of convergence, let's now run the line joining algorithm
        candidates = np.arange(0,len(pot_itcz_coords[0][:]))

        lon_itcz_final = []
        lat_itcz_final = []
        id_final = []

        cnt=0
        while (len(candidates)>0):

            seg = pot_itcz_coords[0][candidates[0]]
            lon_hold = seg[:,0]
            lat_hold = seg[:,1]

            tmp = []
            for i in candidates[1:]:
                segs = pot_itcz_coords[0][i]
                dlon = np.array([lon1-lon2 for lon1 in lon_hold for lon2 in segs[:,0]])
                dlon[dlon>=180] = dlon[dlon>=180]-360
                dlat = np.array([lat1-lat2 for lat1 in lat_hold for lat2 in segs[:,1]])
                dist = np.sqrt(dlon*dlon + dlat*dlat)
                
#                 dist = np.array([gcdist(lon_hold[x],segs[y,0],lat_hold[x],segs[y,1]) for x in np.arange(0,len(lon_hold)) for y in np.arange(0,len(segs[:,0]))])
                #print(dist.min(),dist.max())
                
        
                if (np.any(dist <= 2.5)):
                    lon_hold = np.append(lon_hold,segs[:,0])
                    lat_hold = np.append(lat_hold,segs[:,1])
                else:
                    tmp.append(i)

            candidates = tmp

            #check if the segment is at least 15-deg wide
            dlon_seg = np.max(lon_hold)-np.min(lon_hold)
            dlat_seg = np.max(lat_hold)-np.min(lat_hold)

            if ( (np.abs(dlon_seg) >= 15.0) ):#& (np.abs(dlon_seg)>np.abs(dlat_seg)) ):
                
                lon_hold = lon_hold-lon_shift
                #sort by increasing longitude
                lat_hold_sorted = [x for _,x in sorted(zip(lon_hold,lat_hold))]
                lon_hold_sorted = np.sort(lon_hold)
                
                plt.plot(lon_hold_sorted,lat_hold_sorted)

                lon_itcz_final.extend(lon_hold_sorted)
                lat_itcz_final.extend(lat_hold_sorted)
                id_seg = np.arange(0,len(lat_hold_sorted))
                id_seg[:] = cnt
                id_final.extend(id_seg)
                cnt = cnt+1
            
            
        #append to big dataset that will be saved as a dataframe
        dum = np.arange(0,len(lat_itcz_final))
        dum[:] = date
        #data['time'].extend(pd.to_datetime(dum))
        data['time'].extend(dum)
        data['lat'].extend(lat_itcz_final)
        data['lon'].extend(lon_itcz_final) 
        data['id'].extend(id_final)

            
        return pd.DataFrame(data)
    
    keep_vars = ['divlo','lat','lon','time']
    ds = (read_vars_for_itcz_id(fname,keep_vars))
    ctim1 = date-pd.to_timedelta(7,unit='D')
    ctim2 = date+pd.to_timedelta(7,unit='D')
    dv = ds.divlo.sel(time=slice(ctim1,ctim2)).mean(dim='time')
    grad_dv = (1.0/REarth)*dv.differentiate(coord='lat')
    lap_dv = (1.0/REarth)*(1.0/REarth)*((dv.differentiate(coord='lat')).differentiate(coord='lat')+(dv.differentiate(coord='lon')).differentiate(coord='lon'))
    itcz_loc = grad_dv.where( (dv<-1.5e-6)  & (lap_dv>0.0)) #& (thw>290.0) )      

    [X,Y] = np.meshgrid(np.rad2deg(ds.lon),np.rad2deg(ds.lat))
    cs = plt.contour(X,Y,itcz_loc,colors='k',levels=[0])
    final_coords_date = line_joining_algorithm(cs.allsegs)

    return final_coords_date
    
def get_itcz_coords_given_date_pdv(fname,date,lon_shift,**kwargs):
    
    REarth = 6358.0
    
    def read_vars_for_itcz_id(fname,keep_vars,ctim1,ctim2,subsample):

        ds = xr.open_dataset(fname).sel(time=slice(ctim1,ctim2))

        #convert lat and lon to radians
        ds['lat'] = xr.ufuncs.deg2rad(ds.lat)
        ds['lon'] = xr.ufuncs.deg2rad(ds.lon)

        #compute wet bulb potential temp
        if ('thw' in keep_vars):
            tk = ds.tk850
            qv = ds.qv850
            rh = ds.rh850
            tc = tk-273.15
            tw = tc*np.arctan(0.151977*(rh+8.313659)**0.5)+np.arctan(tc+rh)-np.arctan(rh-1.676331)+0.00391838*(rh)**(1.5)*np.arctan(0.023101*rh)-4.686035
            ds['thw'] = (tw+273.15)*(1000.0/850.0)**(287.15/1004.0)
        #keep only the variables that we need: tw, u, and v
        ds = ds.drop([v for v in ds.variables if v not in keep_vars])
        
        if (subsample):
            ds = ds.resample(time='D').mean()

        return ds  
   
    def merge(lsts):
        sets = [set(lst) for lst in lsts]

        merged = True
        while merged:
            merged = False
            results = []
            while sets:
                common, rest = sets[0], sets[1:]
                sets = []
                for x in rest:
                    if x.isdisjoint(common):
                        sets.append(x)
                    else:
                        merged = True
                        common |= x
                results.append(common)
            sets = results
        return sets
    
    extra_diags = {
            'itcz_width' : False,
            'itcz_intensity' :False,
            'subsample' : False,
        }

    for keys in kwargs:
            extra_diags[keys] = kwargs[keys]
            
    keep_vars = ['div_925_1000mb','thw','lat','lon','time']
            
    if ( (extra_diags['itcz_width']) | (extra_diags['itcz_intensity']) ):
        keep_vars = keep_vars + ['pr']
        
    ctim1 = date-pd.to_timedelta(7.0,unit='D')
    ctim2 = date+pd.to_timedelta(7.0,unit='D')
    ds = (read_vars_for_itcz_id(fname,keep_vars,ctim1,ctim2,extra_diags['subsample']))
    dv = ds.div_925_1000mb.mean(dim='time')
    thw = ds.thw.mean(dim='time')
    #smooth divergence
    # define parameters for the gaussian filter
    sigma = [2.0, 2.0]
    dv = xr.DataArray(sp.ndimage.filters.gaussian_filter(dv, sigma, mode='constant'),coords=[ds.lat,ds.lon],dims=['lat','lon'])
    grad_dv = (1.0/REarth)*dv.differentiate(coord='lat')
    lap_dv = (1.0/REarth)*(1.0/REarth)*((dv.differentiate(coord='lat')).differentiate(coord='lat')+(dv.differentiate(coord='lon')).differentiate(coord='lon'))
    itcz_loc = grad_dv.where( (dv<-1.5e-6)  & (lap_dv>0.0) & (thw>300.0) )        

    #plot contour lines to get coordinates of 0-contour
    plt.figure()
    [X,Y] = np.meshgrid(np.rad2deg(ds.lon),np.rad2deg(ds.lat))
    cs = plt.contour(X,Y,itcz_loc,colors='k',levels=[0])
    plt.close()
    
    #prepare data to create a dataframe
    segs_data = {
        'id': [],
        'lon':[],
        'lat': []
    }
    cnt = 0
    for segs in cs.allsegs[0]:
        segs_data['lon'].extend((segs[:,0]))
        segs_data['lat'].extend((segs[:,1]))
        dum = np.arange(0,len(segs[:,0]))
        dum[:] = cnt
        segs_data['id'].extend(dum)
        cnt+=1

    #create dataframe
    df = pd.DataFrame(segs_data)
    
    #first pass: connect all segments close to each other
    #as in the studies used for reference, we require a maximum distance of 3.5 degrees
    
    #get maximum lon, minimum lon, and average latitude from each segment
    max_lon = df.groupby('id').max().lon
    min_lon = df.groupby('id').min().lon
    avg_lat = df.groupby('id').mean().lat
    lat_at_max_lon = df.lat.iloc[df.groupby('id').idxmax().lon]
    lat_at_min_lon = df.lat.iloc[df.groupby('id').idxmin().lon]
        
    #get the distance between min/max of each segment
    dlon = np.array([(x1-x2) for x1 in max_lon for x2 in min_lon])
    #dlat = np.array([(x1-x2) for x1 in avg_lat for x2 in avg_lat])
    dlat = np.array([(x1-x2) for x1 in lat_at_max_lon for x2 in lat_at_min_lon])
    indx = np.array([[x1,x2] for x1 in df.id.unique() for x2 in df.id.unique()])
    indx2merge = indx[ ((np.abs(dlon)<=3.5) | (np.abs(dlon)>=345.0)) & (np.abs(dlat)<=3.5)]  

    #find are close to each other
    result = merge(indx2merge)
    #print(result)
    
    #merge close segments
    for r in result:
        for s in list(r):
            df.loc[(df.id == s),'id']=np.min(list(r))   

    
    #last pass: eliminate segments that are shorter than 15 degrees and that begin outside the tropics
    #recalculate max_lon and min_lon
    
    max_lon = df.groupby('id').max().lon
    min_lon = df.groupby('id').min().lon
    min_lat = df.groupby('id').min().lat
    id_length_req = df.id.unique()[np.where( (np.abs(max_lon-min_lon)>=15.0) & (min_lat<=5.0))[0]]
    df = df.loc[df['id'].isin(id_length_req)]
    msg = ('found %i segment(s)'%(len(df.id.unique())))
#     sys.stdout.write(msg)
#     sys.stdout.flush()
    
    #two last steps: add time and subtract central longitude
    df['lon'] = df.lon-lon_shift
    
    dum = np.arange(0,len(df))
    dum[:] = date
    df['time'] = dum
    
    #plot for a final check -- FINGERS CROSSED!
#     for i in df.id.unique():
#         df_sub = df[df.id==i].sort_values(by='lon')
#         plt.plot(df_sub.lon,df_sub.lat,'*')

    if ( (extra_diags['itcz_intensity']) ):
        pr = ds.pr.mean(dim='time')*(24.0) #no need to divide over dt
        sigma=2.0
        pr = xr.DataArray(sp.ndimage.filters.gaussian_filter(pr, sigma, mode='constant'),coords=[ds.lat,ds.lon],dims=['lat','lon'])
#         plt.figure()
#         plt.contourf(pr)
#         plt.colorbar()
        itcz_int = []
        for index, row in df.iterrows():
            itcz_int.append(pr.sel(lon=np.radians(row['lon']),lat=np.radians(row['lat']),method='nearest'))
        itcz_int = np.array(itcz_int)
        df['itcz_intensity'] = itcz_int
        
    if ( (extra_diags['itcz_width']) ):
        pr = ds.pr.mean(dim='time')*(24.0) #np need to divide over dt
        
        sigma=2.0
        pr = xr.DataArray(sp.ndimage.filters.gaussian_filter(pr, sigma, mode='constant'),coords=[ds.lat,ds.lon],dims=['lat','lon'])
#         plt.figure(figsize=(16,10))
#         [X,Y] = np.meshgrid(pr.lon,pr.lat)
#         plt.contourf(X,Y,pr,cmap='Blues')
#         plt.colorbar()
#         plt.contour(X,Y,pr,levels=[5.0],colors='k')
        
        pr = pr.where( (pr.lat>=np.radians(-20)) & (pr.lat<=np.radians(20)) )#& (pr>=1.0))
        dlat = pr.lat[1]-pr.lat[0]
        itcz_wth = np.empty(len(df))
        itcz_wth[:] = np.NaN
        cnt = 0
        for index, row in df.iterrows():
            pr1d = pr.sel(lon=np.radians(row['lon']),method='nearest')#.dropna(dim='lat')
            a = pr1d>=2.5
            runsum = a.cumsum()-a.cumsum().where(~a).ffill(dim='lat').fillna(0).astype(int)
            max_lat = runsum[runsum==runsum.max()].lat
            min_lat = max_lat-(runsum.max()*dlat)
#             plt.plot(np.radians(row['lon']),np.radians(row['lat']),'*r')
#             plt.figure()
#             pr1d.plot()
#             break
#             test = np.isnan(pr1d).all()
#             if (not test):
#                 itcz_wth.append( (pr1d.lat.max()-pr1d.lat.min())*6378.0 )
            itcz_wth[cnt] = np.degrees(np.abs(max_lat-min_lat))
            cnt +=1
        if (len(itcz_wth)>1):
            df['itcz_width'] = itcz_wth
        else:
            print(date)
    return df

def get_sinuosity(df):

    gcdist_seg = []
    lenab_seg = []
    for i in df.id.unique():
        df_sub = df[df.id==i].sort_values(by='lon')
        dx = df_sub.lon.diff()
        #check if we're going around the cyclic point
        if (dx.max() > 180.0): 
            #recalculate
            lon = df_sub['lon']
            lon[lon>=df_sub.lon.loc[dx.idxmax()]] = lon[lon>=df_sub.lon.loc[dx.idxmax()]]-360.0
            df_sub['lon'] = lon
            df_sub=df_sub.sort_values(by='lon')
            dx = df_sub.lon.diff()
        dy = df_sub.lat.diff()
        dist = (dx**2+dy**2)**0.5
        gcdist_seg.append(dist.sum())
        lenab_seg.append(df_sub.lon.max()-df_sub.lon.min())
        #something that *could be* improved: interpolating latitudes of different segments
    return (np.sum(gcdist_seg)/np.sum(lenab_seg))

def update_progress(job_title, progress):
    length = 40
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block),
    round(progress*100, 2))
    if progress >= 1: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()

def find_tcgen(data_frames):
    ng = 0
    n = len(data_frames)
    gen_info = []
    for i in range(0,n):
        df = data_frames[i][1]
        #calculate a running mean
        df['pmin'] = df.pmin.rolling(4,min_periods=4).mean()

        dpdt = df.pmin.diff()
        a = dpdt<-0.25
        #find no. consecutive time steps w/ pressure falls
        df['test'] = a.cumsum()-a.cumsum().where(~a).ffill().fillna(0).astype(int)
        #we have a TC when we have surface pressure falls for at least 24 h, that's 4 counts
        df_tc = df[df.test>=1] #originally 4!
    
        if (df_tc.empty):
            print('no genesis time found')
            ng +=1
        else:
            #subtract 18 h to find first intance of pressure falls
            #why 18 and not 24 h? bc four time steps would be 0, 6, 12, 18
            fhr_gen = df_tc.fhr.iloc[0]#-18.0
            #finally we have the info at genesis time
            gen_info.append(df[df.fhr==fhr_gen])
    
    tcgen_info = pd.concat(gen_info)
    print('no genesis: ',len(tcgen_info))

    #as a last step, eliminate those points where a mature TC was detected at genesis time
    tcgen_info = tcgen_info[ (tcgen_info.pmin>=990.0) ]#& (tcgen_info.lat<=30.0)]

    return tcgen_info

def find_nondev_analogs_to_devtcs(pth,tcgen_info):
    """ This function finds analog non-developers to each developing TC.
    An analog is defined as a TC seed with similar vorticity and at a similar
    latitude as a developing TC. The similarity is established by minimizing a
    cost function.
    """
       
    def read_TRACK_write_dataframe(fname):
        cnames=['fhr','lon_TRACK','lat_TRACK','vor850'] 
        data_frames = []
        cnt = 0
        with open(fname) as f:
            for k, v in groupby(islice(f, 3, None),key=lambda x:  x.strip()[0:1].isdigit()):
                val = list(v)
                if k:
                    #if we wanted to save to data frame, we'd uncomment the line below
                    #df = pd.DataFrame(map(str.split,val),columns=cnames,dtype='float')
                    data = np.array(
                        [[float(y) for y in x] for x in[t[0:4] for t in [s for s in [r.split(' ') for r in val]]]]
                    )
                    if ((np.min(data[:,2]) > -10.0 ) & (np.min(data[:,2]) < 35.0) ):
                        cnt += 1
                        df = pd.DataFrame(data,columns=cnames,dtype='float')
                        df['ID'] = saved_track_id
                        df['fhr'] = (df['fhr']+1.0)*6.0
                        data_frames.append(df)
                elif val[-1] == 'Top Total\n':
                    break
                else:
                    saved_track_id = (int(val[0].split()[1]))
        print('found %s cyclone tracks'%cnt)

        return pd.concat(data_frames).reset_index(drop=True)
    
    data_frames_tcs = read_TRACK_write_dataframe("%s/ff_trs_pos.addvor.tcident.35N1tstepwarmcore"%pth)
    data_frames_all = read_TRACK_write_dataframe("%s/ff_trs_pos.addvor"%pth)
        
    ids_all = data_frames_all.ID.unique()
    print('candidates: %i'%len(ids_all))
    ids_tcs = data_frames_tcs.ID.unique()
    print('developing: %i'%len(ids_tcs))

    ids_ndv = [id for id in ids_all if id not in ids_tcs]
    print('non-developing: %i'%len(ids_ndv))  
        
    #adapt to the nondeveloping group only
    data_frames_ndv = data_frames_all[data_frames_all.ID.isin(ids_ndv)]

    #limit to only disturbances first tracked <= 30 N & lasting at least 2 days
    first_lat_ndv = data_frames_ndv.groupby("ID").first()
    first_lat_ndv = first_lat_ndv[first_lat_ndv.lat_TRACK < 30.0]
    ids_ndv = first_lat_ndv.index
    data_frames_ndv =  data_frames_ndv[ data_frames_ndv.ID.isin(ids_ndv)]
    data_frames_ndv = data_frames_ndv.loc[data_frames_ndv.groupby("ID")["fhr"].transform(sum) >= 8]
    ids_ndv = data_frames_ndv.ID.unique()
        
    #loop to find analogs based on latitude and vorticity
    analogs = []
    ids_tcs_gen = tcgen_info.ID.unique()
    print(ids_tcs_gen)
    print(data_frames_tcs.ID.unique())
    vor850=[]
    for i in ids_tcs_gen:
        #vor_tc = tcgen_info[tcgen_info.ID==i].vor850
        lat_tc = tcgen_info[tcgen_info.ID==i].lat_TRACK
        # --- find vor850 out of the original data
        # --- round needed to make sure significant digits don't change
        df_tc = data_frames_tcs[ (data_frames_tcs.ID==i) & (data_frames_tcs.lat_TRACK==round(lat_tc.values[0],6))]
        if (not df_tc.empty):
            vor_tc=df_tc['vor850']
            vor850.append(vor_tc.values[0])
        else:
            print(i)

        vor_ndv = data_frames_ndv.vor850
        lat_ndv = data_frames_ndv.lat_TRACK

        data_frames_ndv['f'] = ((vor_ndv.values-vor_tc.values)/vor_tc.values)**2+0.1*((lat_ndv.values-lat_tc.values)/lat_tc.values)**2

        match = data_frames_ndv[data_frames_ndv.f==data_frames_ndv.f.min()]

        analogs.append(match)

        #eliminate this case altogether
        ids_ndv = [n for n in ids_ndv if n not in list(match.ID)]
        data_frames_ndv = data_frames_ndv[data_frames_ndv.ID.isin(ids_ndv)]   
            
    analogs = pd.concat(analogs)
    tcgen_info['vor850']=vor850
    return analogs,data_frames_all

def find_tcgen_closedcontour(data_frames,ncfile):
    
    def check_closedcountour(ncfile,dLon,dLat,varname,df_sub):
        
        fhr_gen = -999.9
        
        #read netcdf file
        ds = xr.open_dataset(ncfile)
        obsTime = ds.time
                                    
        #drop unnecessary variables
        ds = ds.drop([v for v in ds.variables if v not in ['time','lat','lon',varname]])
        dx = np.abs(ds.lon[1].values-ds.lon[0].values)
   
        lon_array_target = xr.DataArray(np.arange(-dLon,dLon+dx,dx))
        lat_array_target = xr.DataArray(np.arange(-dLat,dLat+dx,dx))
    
        #loop through data, obtain datetime object for valid time, and get corresponding index
        d = 0
        for index, row in df_sub.iterrows():
            
            #define variable to check if we have found a genesis time
            closed = False
            
            #get time, lat, and lon of TC point
            valid_time = pd.to_datetime(row.valid_time,format='%Y-%m-%d_%H.%M.%S')
            clon = row['lon']
            clat = row['lat']
     
            #get fields centered on the lat and lon of the TC (or disturbance) center
            var2tst = ds[varname].sel(time=valid_time).sortby('lat')
            var2tst['lon'] = var2tst.lon-ds.lon.sel(lon=clon,method='nearest')
            var2tst['lat'] = var2tst.lat-ds.lat.sel(lat=clat,method='nearest')
        
            #check if we're east or west of Greenwich, and adjust the longitude array
            if (var2tst.lon.max() < lon_array_target.max()):
                #need to find lon<-180 and add 360
                lon_temp = var2tst.lon.where(var2tst.lon>-180.0,var2tst.lon+360.0)
                #reassign longitude coordinate
                var2tst['lon'] = lon_temp
                #pivot around longitude 0 by finding longitudes <=0 and those > 0
                group1 = var2tst.where( (var2tst.lon > -180.0) & (var2tst.lon <= 0.0), drop=True )
                group2 = var2tst.where( (var2tst.lon > 0.0) & (var2tst.lon <= 180.0), drop=True )
                #make sure that the coordinate is in ascending order
                group2 = group2.sortby(group2.lon)
                del(var2tst)
                #finally merge the two!
                var2tst = xr.concat([group1,group2],dim='lon')
            elif (var2tst.lon.min() > lon_array_target.min() ):
                lon_temp = var2tst.lon.where(var2tst.lon<180.0,var2tst.lon-360.0)
                var2tst['lon'] = lon_temp
                group1 = var2tst.where( (var2tst.lon > -180.0) & (var2tst.lon <= 0.0), drop=True )
                group2 = var2tst.where( (var2tst.lon > 0.0) & (var2tst.lon <= 180.0), drop=True )
                group1 = group1.sortby(group1.lon)
                group2 = group2.sortby(group2.lon)
                del(var2tst)
                var2tst = xr.concat([group1,group2],dim='lon')
            
            # finally get the TC-centered field
            var_intrp = var2tst.sel(lon=slice(-(dLon+dx/2),dLon+dx/2),lat=slice(-(dLat+dx/2),dLat+dx/2))
            
            #smooth out before we plot w/ a gaussian filter -- using sigma=10 bc of high res data
            var_intrp_smooth = sp.ndimage.filters.gaussian_filter(var_intrp, 5.0)

            # make a dummy plot; we'll use this graphic object to get the coordinates of the MSLP contours
            [X,Y] = np.meshgrid(var_intrp.lon,var_intrp.lat)
            fig=plt.figure()
            levels=np.arange(var_intrp_smooth.min(),var_intrp_smooth.max()+100,100)
            cs = plt.contour(X,Y,var_intrp_smooth,colors='k',levels=levels) #this is our object
            
            # check for closed contours if there are at least 2 contour levels (otherwise the data are all uniform)
            if (len(levels) > 2):
                #loop over each contour
                for kk in np.arange(0,len(levels)):
                    # get the segments from each contour
                    for segs in cs.allsegs[kk]:
                        # make sure that each segment is long enough (i.e., not just a line from one point to the next)--- 5 points works as a good threshold
                        if (len(segs[:,0]) > 5):
                            # use PyNGL library to check if the contour is a circle
                            test = Ngl.gc_inout(0.0,0.0,segs[:,1],segs[:,0]) #output should be 1 if it's a circle
                            # if the contour is a circle, check that the first and last point are close to each other 
                            if ( (test==1) & ( np.abs(segs[0,0]-segs[-1,0]) <= dx ) & ( np.abs(segs[0,1] - segs[-1,1]) <= dx)):
                                # we've found a closed contour!
                                plt.plot(segs[:,0],segs[:,1],'.-r')
                                plt.title(df.ID.unique())
                                # update variable so we exit the loop
                                closed = True
                                break
                    if (closed):
                        break
            
            d += 1

            if (closed):
                # this is our genesis time
                fhr_gen = row['fhr']
                display(fig)
                plt.close()
                break
            else:
                plt.close()
                
        if (fhr_gen == -999.9):
            # plot again to show
            fig=plt.figure()
            levels=np.arange(var_intrp_smooth.min(),var_intrp_smooth.max()+100,100)
            cs = plt.contour(X,Y,var_intrp_smooth,colors='k',levels=levels) #this is our object
            plt.title('no closed contour')
            display(fig)
            plt.close()
        
        return fhr_gen
            
    #define basic variables        
    ng = 0
    n = len(data_frames)
    gen_info = []
    
    #loop over each TC == each data frame
    for i in range(0,n):
        df = data_frames[i][1]
        #calculate a running mean
        df['pmin'] = df.pmin.rolling(4,min_periods=4).mean()

        dpdt = df.pmin.diff()
        a = dpdt<0.0
        #find no. consecutive time steps w/ pressure falls
        df['test'] = a.cumsum()-a.cumsum().where(~a).ffill().fillna(0).astype(int)
        #we have a TC when we have surface pressure falls for at least 24 h, that's 4 counts
        df_tc = df[df.test>=4]
        #RRB:1/30/2023
        #This procedure checks for a falling pressure for 4 consecutive time steps, which means that
        #we need to go back 4 time steps to make sure that we account for the initial pressure drop.
        
        if (df_tc.empty):
            print('no pressure drop found')
            fig=plt.figure()
            plt.plot(df.fhr,df.pmin)
            display(fig)
            plt.close()
            
            # we didn't find falling pressure, but we will still check for a closed circulation
            fhr_gen = check_closedcountour(ncfile,5*2.25,5*2.225,'mslp',df) #using df instead of df_tc
            ng +=1
        else:
            #let's redefine the data frame to account for the times when the pressure begins to fall
            df_tc_red = []
            for index, row in df_tc.iterrows():
                if (row['test']==4):
                    # going 12 hours back in time to catch the 'center time' of when pressure is falling
                    df_tc_red.append(df[ (df.fhr>=row['fhr']-12) & (df.fhr <= row['fhr'])] )
                else:
                    df_tc_red.append(df_tc[df_tc.fhr==row['fhr']])
            # concatenate into a single data frame
            df_tc_red = pd.concat(df_tc_red)
            del(df_tc)
            
            fig=plt.figure()
            plt.plot(df.fhr,df.pmin)
            plt.plot(df_tc_red.fhr,df_tc_red.pmin)
            display(fig)
            plt.close()
            
            #we have found falling surface pressure; let's check if there's a closed contour
            fhr_gen = check_closedcountour(ncfile,5*2.25,5*2.225,'mslp',df_tc_red)
                                        
            if (fhr_gen != -999.9):
                gen_info.append(df[df.fhr==fhr_gen])
            else:
                print('no closed contour found in MSLP')
    
    tcgen_info = pd.concat(gen_info)
    print('no genesis: ',len(tcgen_info))

    #as a last step, eliminate those points where a mature TC was detected at genesis time
    tcgen_info = tcgen_info[ (tcgen_info.pmin>=990.0) ]#& (tcgen_info.lat<=30.0)]

    return tcgen_info

def filter_data_for_ccews(fname,obsVar,obsPerDay,waveName,**kwargs):
    
    filter_info = {
        'minLat' : 0,
        'maxLat' : 10,
        'avgLat' : True
    }
    
    for keys in kwargs:
        filter_info[keys] = kwargs[keys]
    print(filter_info)
    
    #read data    
    infile = xr.open_dataset(fname)
    lon = infile.lon

    obsTime = infile.time
    nObs=len(obsTime)
    obsData = infile[obsVar].sel(lat=slice(filter_info['minLat'],filter_info['maxLat']))#.mean(dim='lat')
    
    from datetime import timedelta
    # Extend filtering to 700 days
    filtTime = np.arange(0,len(obsTime)+702)

    # Incorporate original data into array
    if (filter_info['avgLat']):
        obsData = obsData.mean('lat')
        filtData=np.zeros([len(filtTime),1,len(lon)],dtype='f')
        lat = [0]
        coords = [obsTime,lon]
        dims = ['time','lon']
        filtData[0:obsData.shape[0],0,:]=obsData  
    else:
        lat = obsData.lat
        filtData=np.zeros([len(filtTime),len(lat),len(lon)],dtype='f')
        coords = [obsTime,lat,lon]
        dims = ['time','lat','lon']
        filtData[0:obsData.shape[0],:]=obsData  
    
    wave = filtData
    
    if (waveName == 'Kelvin'):
        wave_longname="Kelvin Waves"
        wave_filter="Straub & Kiladis (2002) to 20 days"
        wave_wavenumber=np.array([1,14],dtype='f')
        wave_period=np.array([2.5,20],dtype='f')
        wave_depth=np.array([8,90],dtype='f')
        # wave_units=unit
    elif (waveName == 'AD'):
        wave_longname="Advective Disturbances"
        wave_filter="Straub & Kiladis (2002) to 20 days"
        wave_wavenumber=np.array([-28,-4],dtype='f')
        wave_period=np.array([5.0,20],dtype='f')
        wave_depth=np.array([-999,-999],dtype='f') 
    elif (waveName == 'TD'):
        wave_longname="TDs"
        wave_filter="Straub & Kiladis (2002) to 12 days"
        wave_wavenumber=np.array([-28,-6],dtype='f')
        wave_period=np.array([2.5,12],dtype='f') #12 days following Kiladis et al.
        wave_depth=np.array([-999,-999],dtype='f')
    elif (waveName == 'MRG'):
        wave_longname="MRGs"
        wave_filter=""
        wave_wavenumber=np.array([-10,-1],dtype='f')
        wave_period=np.array([3,10],dtype='f') 
        wave_depth=np.array([8,90],dtype='f')        

    #################################################
    # Filter 
    #################################################
    print("\n##############################\nFiltering %s\n"%waveName)
    for lat_counter in range(0,len(lat)):
        wave[:,lat_counter,:]=kf_filter(filtData[:,lat_counter,:],obsPerDay,\
                       wave_period[0],wave_period[1],\
                       wave_wavenumber[0],wave_wavenumber[1],\
                       wave_depth[0],wave_depth[1],waveName)
        
    waveData = xr.DataArray(wave[0:len(obsTime),:].squeeze(),coords=coords,dims=dims)
        
    return waveData

def filter_data_for_ccews_YHSmethod(fname,uname,vname,zname,obsPerDay,waveName,**kwargs):

    filter_info = {
        'minLat' : -20,
        'maxLat' :  20,
        'nTaper' : 100,
    }
    
    def zeroPad(varIn,nTaper):

        extendedTime = np.arange(0,len(varIn.time)+(nTaper*2*obsPerDay))
        nTimeExt = len(extendedTime)

        varPadded = np.zeros((len(varIn.lat),len(varIn.lon),len(extendedTime)))
        varPadded[:,:,nTaper*obsPerDay:nTaper*obsPerDay+len(varIn.time)]=varIn.transpose('lat','lon','time')

        return varPadded
    
    #read data    
    infile = xr.open_dataset(fname).sel(lat=slice(filter_info['minLat'],filter_info['maxLat']),time=slice('2000-03-31','2000-08-11'))
    lon = infile.lon
    lat = infile.lat
    obsTime = infile.time
    
    uwind = infile[uname]
    vwind = infile[vname]
    
    #geopotential height is constant for now
    if ('850' in uname):   
        Z = uwind.copy()
        Z[:] = 1528.128
    elif ('200' in uname):
        Z = uwind.copy()
        Z[:] = 11941.85
    else:
        print('wrong level')

    # Pad with zeroes at the beginning and end. 
    # This is left outside the function such that the user can decide on whether to pad or not and by how many points.
    nTaper = filter_info['nTaper']
    uPadded = zeroPad(uwind,nTaper)
    vPadded = zeroPad(vwind,nTaper)
    zPadded = zeroPad(Z,nTaper)
    
    # define limits for the filter
    tMin = 2.0
    tMax = 30.0
    if (waveName == 'Kelvin'):
        kMin = 2.0
        kMax = 40.0
    elif (waveName == 'ER1') | (waveName == 'IG1') | (waveName == 'WMRG'):
        #westward moving waves; i.e., negative wavenumbers
        kMin = -40.0
        kMax = -2.0

    # we are ready to call the function that does the filtering, projection, and recovery
    uWave, vWave, zWave = kf_filter_plus_PCF_projection(uPadded,vPadded,zPadded,lat,obsPerDay,tMin,tMax,kMin,kMax,waveName)    
    
    # as a final step, convert uWave, vWave, and ZWave to xarray data arrays and remove the taper
    # coordinates are flipped
    dims = ['lat','lon','time']
    coords = [lat,lon,obsTime]
    uWaveOut = xr.DataArray(
        uWave[:,:,nTaper*obsPerDay:-nTaper*obsPerDay],dims=dims,coords=coords,
    ).transpose('time','lat','lon')
    
    vWaveOut = xr.DataArray(
        vWave[:,:,nTaper*obsPerDay:-nTaper*obsPerDay],dims=dims,coords=coords,
    ).transpose('time','lat','lon')
    zWaveOut = xr.DataArray(
        zWave[:,:,nTaper*obsPerDay:-nTaper*obsPerDay],dims=dims,coords=coords,
    ).transpose('time','lat','lon')
    
    return uWaveOut,vWaveOut,zWaveOut

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def format_valid_time(dfIn):
    time = []
    for index, row in dfIn.iterrows():
        ctim = np.datetime64(pd.to_datetime(row['valid_time'],format='%Y-%m-%d_%H.%M.%S'))
        time.append(ctim)
    dfIn['time']=time
    return dfIn

def get_plot_waverelative_hovmoller(pth,expName,waveName,minLat,maxLat,list_of_vars,**kwargs):

    if 'panelLabels' not in kwargs:
        kwargs['panelLabels'] = False

    # set global parameters -- should these be input to the function?
    dLon = 30.0
    nDays = 7

    #increase font size
    plt.rcParams.update({'font.size': 18})

    fname = pth+expName+'/latlon/diags_global_nospinup_r1440x720.nc'

    if ('allTCG' in waveName):
        # find the TCG locations
        tcgen_info = pd.read_csv('TCG_TC_3km.csv')
        tcgen_info = tcgen_info[tcgen_info.lat <= 20]
        lon = tcgen_info.lon.values
        time = []
        for index, row in tcgen_info.iterrows():
            ctim = np.datetime64(pd.to_datetime(row['valid_time'],format='%Y-%m-%d_%H.%M.%S'))
            time.append(ctim)
        # number of points to skip
        skip=1
    elif (waveName=='KelvinTCG'):
        tcgen_info = pd.read_csv('Kelvin_TCG_on_peak.csv')#,parse_dates=['time'])
        tcgen_info = tcgen_info[tcgen_info.lat <= 20]
        lon = tcgen_info.lon.values
        time = []
        for index, row in tcgen_info.iterrows():
            ctim = np.datetime64(pd.to_datetime(row['valid_time'],format='%Y-%m-%d_%H.%M.%S'))
            time.append(ctim)
        # number of points to skip
        skip=1
    elif (waveName =='notKelvinTCG'):
        tcgen_info = pd.read_csv('Kelvin_TCG_not_on_peak.csv')#,parse_dates=['time'])
        tcgen_info = tcgen_info[tcgen_info.lat <= 20]
        lon = tcgen_info.lon.values
        time = []
        for index, row in tcgen_info.iterrows():
            ctim = np.datetime64(pd.to_datetime(row['valid_time'],format='%Y-%m-%d_%H.%M.%S'))
            time.append(ctim)
        # number of points to skip
        skip=1
    elif ('TDTCG' in waveName):
        tcgen_info = pd.read_csv('TD_TCG_on_peak.csv')#,parse_dates=['time'])
        tcgen_info = tcgen_info[tcgen_info.lat <= 20]
        lon = tcgen_info.lon.values
        time = []
        for index, row in tcgen_info.iterrows():
            ctim = np.datetime64(pd.to_datetime(row['valid_time'],format='%Y-%m-%d_%H.%M.%S'))
            time.append(ctim)
        # number of points to skip
        skip=1
    else:
        #check if I have already saved the filtered wave data
        waveFile = pth+expName+'/latlon/'+waveName+'_r1440x720.csv'
        if os.path.exists(waveFile):
            df = pd.read_csv(waveFile)
            lon=df.lon.values
            time=[]
            for index, row in df.iterrows():
                ctim = np.datetime64(pd.to_datetime(row['time'],format='%Y-%m-%d %H:%M:%S'))
                time.append(ctim)
        else:
            fname = pth+expName+'/latlon/diags_global_nospinup_r1440x720.nc'
            lon, time, _, _ = find_wave_peak_graphically(fname,waveName)
            #save file for future use
            data2save = {
                'lon':  lon,
                'time': [t.values for t in time]
            }
            df=pd.DataFrame(data2save)
            df.to_csv(waveFile,index=False)
        skip=1
        # lon, time, waveData, waveDatadt = find_wave_peak_graphically(fname,waveName)
        # # normalize
        # waveData = waveData/waveData.std(dim=('time'))
        # waveDatadt = waveDatadt/waveDatadt.std(dim=('time'))
        # waveData = waveData.where( (waveData**2.0+waveDatadt**2.0)**0.5 >= 1.0 )


    waveLons = lon[0:len(lon):skip] #choose only some waves for now; randomnized later on
    waveTime = time[0:len(time):skip]
    print('no. wave peaks: %i'%len(waveLons))

    output = {}
    for varName in list_of_vars:
        print(varName)

        #check if we need filtered data for a second wave
        if ('filtered' in varName):

            if (kwargs['waveName2'] != waveName):
                _, _, waveData2, waveDatadt2 = find_wave_peak_graphically(fname,kwargs['waveName2'])
                # normalize
                print(waveData2.min(),waveData2.max())
                print('NORMALIZING')
                waveData2 = waveData2/waveData2.std(dim=('time'))
                print(waveData2.min(),waveData2.max())
                # waveDatadt2 = waveDatadt2/waveDatadt2.std(dim=('time'))
            else:
                _, _, waveData, waveDatadt = find_wave_peak_graphically(fname,kwargs['waveName2'])
                # normalize
                print('NORMALIZING')
                waveData = waveData/waveData.std(dim=('time'))
                # waveDatadt = waveDatadt/waveDatadt.std(dim=('time'))
                waveData2 = waveData

            # waveData2 = waveData2.where( (waveData2**2.0+waveDatadt2**2.0)**0.5 >= 1.0 )


        ds = xr.open_dataset(fname).sel(lat=slice(minLat,maxLat))

        dx = np.abs(ds.lon[1].values-ds.lon[0].values)

        lon_array_target = xr.DataArray(np.arange(-dLon,dLon+dx,dx))
        tim_array_target = xr.DataArray(np.arange(-nDays,nDays+0.25,0.25))

        centered_data_0s = np.empty( (len(waveLons),len(tim_array_target),len(lon_array_target)) )
        centered_data_0s[:] = np.nan
        IDs = np.arange(0,len(waveLons))

        ds_out = xr.DataArray(centered_data_0s,dims=('ID','time','lon'),coords=(IDs,tim_array_target,lon_array_target))

        if ('filtered' in varName):
            if kwargs['waveName2'] == waveName:
                varAllTimes = waveData
            else:
                # lon2, time2, waveData = find_wave_peak_graphically(fname,kwargs['waveName2'])
                varAllTimes = waveData2
        else:
            if (varName == 'SAT_FRACTION'):
                varAllTimes = (ds.qv_vint/ds.qvs_vint).mean('lat')
            elif (varName == 'SHEAR_VOR'):
                Rearth = 6379.0e3
                uwind = ds.u850
                vwind = ds.v850
                U = (uwind**2.0 + vwind**2.0)**0.5
                x = np.radians(U.lon)*Rearth
                y = np.radians(U.lat)*Rearth
                dudx = np.gradient(uwind,x,axis=2)
                dudy = np.gradient(uwind,y,axis=1)
                dvdx = np.gradient(vwind,x,axis=2)
                dvdy = np.gradient(vwind,y,axis=1)
                varAllTimes = ((U**(-2.0))*( (vwind*uwind*dudx) +\
                                            (vwind*vwind*dvdx) -\
                                            (uwind*uwind*dudy) -\
                                            (uwind*vwind*dvdy))).mean('lat')
            elif (varName == 'CURVATURE_VOR'):
                Rearth = 6379.0e3
                uwind = ds.u850
                vwind = ds.v850
                U = (uwind**2.0 + vwind**2.0)**0.5
                x = np.radians(U.lon)*Rearth
                y = np.radians(U.lat)*Rearth
                dudx = np.gradient(uwind,x,axis=2)
                dudy = np.gradient(uwind,y,axis=1)
                dvdx = np.gradient(vwind,x,axis=2)
                dvdy = np.gradient(vwind,y,axis=1)
                varAllTimes = ((U**(-2.0))*( (uwind*uwind*dudx) -\
                                            (vwind*vwind*dudy) -\
                                            (uwind*uwind*dudx) +\
                                            (uwind*vwind*dvdy))).mean('lat')
            elif ('VI' in varName) | ('SHRM' in varName) |\
                     ('CHI_TE' in varName) | ('MPI_VMAX' in varName) | \
                     ('AIRSEA' in varName) | ('CHI_NUM' in varName) | \
                     (varName == 'RHUM') | ('VOR' in varName) | ('DIV' in varName):
                del(ds)
                ds = xr.open_dataset(pth+expName+'/latlon/genesis_indexes_r3600x1800_revised_04042023.nc').sel(lat=slice(minLat,maxLat))
                if ('CHI_NUM' in varName):
                    ds['CHI_NUM'] = ds.SPMS-ds.SPM
                    print(ds)
                if ('log' in varName):
                    tmpVar = ds[varName.split('log')[1]]
                    tmpVar = tmpVar.where(np.fabs(tmpVar) > 0.0)
                    varAllTimes = np.log(tmpVar).mean('lat')
                    if ('MPI' in varName): #we need to multiply by -1 bc -ln(MPI)
                        varAllTimes = -1.0*varAllTimes
                else:
                    varAllTimes = ds[varName].mean('lat')
            elif (varName == 'GPI_E'):
                del(ds)
                ds = xr.open_dataset(pth+expName+'/latlon/genesis_indexes_r360x180.nc').sel(lat=slice(minLat,maxLat))
                print(ds.lat.min(),ds.lat.max())
                #calculate the GPI from Emanuel (2010), except using CHI from the ventilation index
                MPI = ds.MPI_VMAX
                CHI = ds.CHI_TE
                VOR = ds.VOR850
                # # smooth out vorticity
                # sigma=5.0
                # VOR = xr.DataArray(sp.ndimage.filters.gaussian_filter(VOR, sigma, mode='constant'),coords=VOR.coords,dims=VOR.dims)
                # # make test plot
                # fig=plt.figure()
                # plt.contourf(VOR.mean('time'))
                # display(fig)
                # we need absolute vorticity
                fcor = 2.0*7.292e-5*np.sin(np.radians(VOR.lat))
                VOR = VOR+fcor
                SHR = ds.SHRM
                # restrict to positive vorticity and non-zero CHI_TE
                VOR = VOR.where(VOR>0.0)
                CHI = CHI.where(CHI>0.0)
                # restrict to MPI - 35 > 0
                MPI = MPI-35.0
                MPI = MPI.where(MPI>0.0,0.0)
                # add 25 m/s to shear
                SHR = SHR+25.0
                GPI = ((np.fabs(VOR)**3.0) * (CHI**(-4.0/3.0)) * (MPI**3.0))/(SHR**4.0)
                varAllTimes = GPI.mean('lat')
                del(VOR,SHR,MPI,CHI)
            elif ('STRETCHING' in varName):
                del(ds)
                plev = varName[10:13]
                ds = xr.open_dataset(pth+expName+'/latlon/genesis_indexes_r3600x1800_revised_04042023.nc').sel(lat=slice(minLat,maxLat))
                fCor = 2.0*7.292e-5*np.sin(np.radians(ds.lat))
                absVOR = ds['VOR%s'%plev]+fCor
                varAllTimes = -1.0*(absVOR*ds['DIV%s'%plev]).mean('lat')
            elif ('Yang' in varName):
                del(ds)
                fname = pth+expName+'/latlon/diags_gaussian_global_nospinup_r3600x1800.nc'
                ds=xr.open_dataset(fname).sel(lat=slice(minLat,maxLat))
                # we want filtered winds using the YHS method
                plev = varName[1:4]
                uname = 'u'+plev
                vname = 'v'+plev
                # assuming it's the same wave
                print('filtering')
                uWave, vWave, zWave = filter_data_for_ccews_YHSmethod(fname,uname,vname,'z',obsPerDay,kwargs['waveName2'])
                print(uWave.min(),uWave.max())
                if (kwargs['waveName2'] == 'Kelvin'):
                    varAllTimes = uWave.sel(lat=slice(minLat,maxLat)).mean('lat')
                    print(varAllTimes.min(),varAllTimes.max())
                elif (kwargs['waveName2'] == 'WMRG'):
                    varAllTimes = vWave.sel(lat=0,method='nearest')
                elif (kwargs['waveName2'] == 'ER1'):
                    varAllTimes = vWave.sel(lat=8,method='nearest')
                elif (kwargs['waveName2'] == 'ER2'):
                    varAllTimes = vWave.sel(lat=13,method='nearest')
                else:
                    print('decide which variable to plot for other waves')
            elif (varName == 'P2'):
                ds = xr.open_dataset(pth+expName+'/latlon/genesis_indexes_r3600x1800_revised_04042023.nc').sel(lat=slice(minLat,maxLat))
                VI = ds['VI'].mean('lat')
                VI0 = VI.mean('time').mean('lon')
                print(VI0)
                alpha=0.5
                varAllTimes = 100.0*(1.0/( (1.0+(VI0/VI)**(-1.0/alpha)) ))
                print(varAllTimes.min(),varAllTimes.max())
            else:
                varAllTimes = ds[varName].mean('lat')

        #subtract the time and zonal mean
        normalize = False
        if ('filtered' not in varName) & ('Yang' not in varName) & ('log' not in varName):# & ('P2' not in varName):
            print('normalizing')
            stddev = varAllTimes.std()
            if (normalize):
                varAllTimes = (varAllTimes - varAllTimes.mean('time').mean('lon'))/stddev
            else:
                varAllTimes = (varAllTimes - varAllTimes.mean('time').mean('lon'))

        #loop through data, obtain datetime object for valid time, and get corresponding index
        d = 0
        for d in np.arange(0,len(waveLons)):
            clon = waveLons[d]
            #enter code here to check if time and longitude correspond to TCG

            #we'd like to save nx x ny data points, but in some cases the points may be too close to the boundarys
            tim1 = waveTime[d]-pd.to_timedelta(nDays,unit='D')
            tim2 = waveTime[d]+pd.to_timedelta(nDays,unit='D')
            lon1 = clon-dLon
            lon2 = clon+dLon

            #choose variable
            var2tst = varAllTimes.sel(time=slice(tim1,tim2))

            #transform time relative to feature time
            var2tst['time'] = (var2tst.time-waveTime[d])/np.timedelta64(1,'D')
            tim_rel = tim_array_target.where( (tim_array_target>=var2tst.time.min()) & (tim_array_target<=var2tst.time.max()) )

            # check if the arrays in and out have consistent time dimensions
            # if not, we're too close to one of the edges. we'd like to skip this time
            tout1 = tim_rel.argmin().values
            tout2 = tim_rel.argmax().values+1
            if ( (tout2-tout1) == len(var2tst.time)):

                #transform longitude to be centered on the feature
                var2tst['lon'] = var2tst.lon-ds.lon.sel(lon=clon,method='nearest')

                # flip if necessary along the cyclic point
                if (var2tst.lon.max() < lon_array_target.max()):
                    #need to find lon<-180 and add 360
                    lon_temp = var2tst.lon.where(var2tst.lon>-180.0,var2tst.lon+360.0)
                    #reassign longitude coordinate
                    var2tst['lon'] = lon_temp
                    #pivot around longitude 0 by finding longitudes <=0 and those > 0
                    group1 = var2tst.where( (var2tst.lon > -180.0) & (var2tst.lon <= 0.0), drop=True )
                    group2 = var2tst.where( (var2tst.lon > 0.0) & (var2tst.lon <= 180.0), drop=True )
                    #make sure that the coordinate is in ascending order
                    group2 = group2.sortby(group2.lon)
                    del(var2tst)
                    #finally merge the two!
                    var2tst = xr.concat([group1,group2],dim='lon')
                elif (var2tst.lon.min() > lon_array_target.min() ):
                    lon_temp = var2tst.lon.where(var2tst.lon<180.0,var2tst.lon-360.0)
                    var2tst['lon'] = lon_temp
                    group1 = var2tst.where( (var2tst.lon > -180.0) & (var2tst.lon <= 0.0), drop=True )
                    group2 = var2tst.where( (var2tst.lon > 0.0) & (var2tst.lon <= 180.0), drop=True )
                    group1 = group1.sortby(group1.lon)
                    group2 = group2.sortby(group2.lon)
                    del(var2tst)
                    var2tst = xr.concat([group1,group2],dim='lon')

                var_intrp = var2tst.sel(lon=slice(-(dLon+dx/2),dLon+dx/2))

                ds_out[d,tout1:tout2,:] = var_intrp.values
            else:
                print('skipping because we are too close to the beginning or end of the time period')
                print(tout1,tout2,len(var2tst.time))

            # plt.figure()
            # plt.contourf(ds_out[d,:,:])
            # plt.title(d)
            # break
        print(d)

        centered_data_avg = ds_out.mean(dim='ID')
        output[varName] = centered_data_avg
        del(ds_out)

        #subtract time, zonal mean
        #centered_data_avg = my_dict[varname][0]*(centered_data_avg)
        # centered_data_avg = centered_data_avg-centered_data_avg.mean()
        print(centered_data_avg.min(),centered_data_avg.max())

    # #     with open('test_%s_%s.npy'%(exps[n],varname), 'wb') as f:
    # #         np.save(f,centered_data_avg)

        if ('filtered' in varName):
            with open(pth+expName+'/latlon/saved_composites/%s_waves_composite_hovmoller_%s_%s_MPAS_%s_aqua_sstmax10N.npy'%(waveName,varName,kwargs['waveName2'],expName), 'wb') as f:
                np.save(f,centered_data_avg)
        else:
            with open(pth+expName+'/latlon/saved_composites/%s_waves_composite_hovmoller_%s_MPAS_%s_aqua_sstmax10N.npy'%(waveName,varName,expName), 'wb') as f:
                np.save(f,centered_data_avg)

        fig, ax = plt.subplots(figsize=(10,10))
        [X,Y] = np.meshgrid(lon_array_target,tim_array_target)
        # scale by desired factor
        # centered_data_avg = centered_data_avg*my_dict[varName][0]
        cf = ax.contourf(X,Y,centered_data_avg,extend='both')#,cmap=my_dict[varName][1],levels=my_dict[varName][2])
        if (('overlay' in kwargs) & (kwargs['overlay'])):
            #load the file
            with open(pth+expName+'/latlon/saved_composites/%s_waves_composite_hovmoller_%s_%s_MPAS_%s_aqua_sstmax10N.npy'%(waveName,'filteredpr','Kelvin',expName), 'rb') as f:
                centered_data_avg = np.load(f)
            print('overlay!')
            cl1 = ax.contour(X,Y,centered_data_avg,levels=[0.5,1.0,1.5,2.0],colors='white',linewidth=4)
            cl2 = ax.contour(X,Y,centered_data_avg,levels=[0.5,1.0,1.5,2.0],colors='magenta',linewidth=3)
            ax.clabel(cl1, cl1.levels, inline=1, fontsize=18,fmt='%2.2f')
            ax.clabel(cl2, cl2.levels, inline=1, fontsize=18,fmt='%2.2f')

        if ('TCG' in waveName):
            ax.set_xlabel('longitude w.r.t. cyclogenesis')
            ax.set_ylabel('days w.r.t. cyclogenesis')
        elif ('TD' in waveName):
            ax.set_xlabel('longitude w.r.t. easterly wave peak')
            ax.set_ylabel('days w.r.t. easterly wave peak')
        else:
            ax.set_xlabel('longitude w.r.t. %s wave peak'%waveName)
            ax.set_ylabel('days w.r.t. %s wave peak'%waveName)
        # minor ticks every 15 deg
        ax.xaxis.set_major_locator(MultipleLocator(15))
        # minor ticks every 5 deg
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        # major ticks every 1 day
        ax.yaxis.set_major_locator(MultipleLocator(1))
        # minor ticks every 6 h = 0.25 days
        ax.yaxis.set_minor_locator(MultipleLocator(0.25))
        plt.hlines(0,-dLon,dLon)
        plt.vlines(0,-nDays,nDays)
        ax.grid(True)

        cbar_ax = fig.add_axes([0.15, -0.05, 0.8, 0.02])
        cbar=fig.colorbar(cf, cax=cbar_ax,orientation='horizontal')
        # cbar.set_label(my_dict[varName][3])
        fig.tight_layout(pad=0.1)

        # add panel labels if requested
        if ('panelLabels' in kwargs) & (kwargs['panelLabels']):
            pLabel=my_dict[varName][-1]
            ax.text(left, top, pLabel,
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes,
                fontsize='16', bbox=dict(facecolor='white', alpha=0.85))
        display(fig)
        # plt.savefig('./Figures/hovmoller_'+varName+'_wrt_'+waveName+'.png',bbox_inches='tight')
        plt.close()
    return output

def find_wave_peak_graphically(fname,waveName):

    # Set Pre-Defined Arguments for wave filtering
    mis = -999
    obsPerDay = 4
    dt = 6.0
    algoName='mpas'

    obsVar = 'pr'
    filtVar = obsVar
    unit = 'mm hr-1'

    #get latitude ranges for each wave
    print('warning: hard coded latitude bands')
    minLat = -10  #waves[waveName][4]
    maxLat = 10   #waves[waveName][5]

    waves = {
    'Kelvin': [np.array([1,14],dtype='f'),np.array([2.5,20],dtype='f'),np.array([8,90],dtype='f'),[-1.0,31.0],minLat,maxLat],
    'AD': [np.array([-28,-4],dtype='f'),np.array([5.0,20],dtype='f'),np.array([mis,mis],dtype='f'),[-30.0,1.0],minLat,maxLat],
    'IG1': [np.array([-15,-1],dtype='f'),np.array([1.4,2.6],dtype='f'),np.array([8,90],dtype='f'),[-75.0,-0.0],minLat,maxLat],
    'MRG': [np.array([-10,-1],dtype='f'),np.array([3,10],dtype='f'),np.array([8,90],dtype='f'),[-30.0,1.0],minLat,maxLat],
    'IG0': [np.array([1,14],dtype='f'),np.array([1.8,4.5],dtype='f'),np.array([8,90],dtype='f'),[-30.0,1.0],minLat,maxLat],
    'IG2': [np.array([-15,-1],dtype='f'),np.array([1.25,2.0],dtype='f'),np.array([8,90],dtype='f'),[-30.0,1.0],minLat,maxLat],
    'ER': [np.array([-10,-1],dtype='f'),np.array([10.0,40.0],dtype='f'),np.array([8,90],dtype='f'),[-30.0,1.0],minLat,maxLat],
    # 'TD': [np.array([-20,-6],dtype='f'),np.array([2.5,5.0],dtype='f'),np.array([8,90],dtype='f'),[-30.0,1.0],minLat,maxLat]
    #2/9/2022: modified the space-time filter of TDs to be consistent with Kiladis et al. (2006): wavenumbers 6-greater, and frequencies of 2.5-12 days
    'TD':  [np.array([-20,-6],dtype='f'),np.array([2.5,12.0],dtype='f'),np.array([mis,mis],dtype='f'),[-30.0,1.0],minLat,maxLat]

    }
    
    #increase font size
    plt.rcParams.update({'font.size': 14})

    ds = xr.open_dataset(fname)

    obsData = ds.pr.sel(lat=slice(minLat,maxLat)).mean(dim='lat')

#     #subtract time mean
#     print('calculating and subtracting climo...')
#     obsData = obsData-obsData.mean(dim='time')
#     print('done w/ climo')

    lon = obsData.lon
    obsTime = obsData.time
    nObs=len(obsTime)

    # Extend filtering to 700 days
    filtTime = np.arange(0,len(obsTime)+702)

    # Incorporate original data into array
    filtData=np.zeros([len(filtTime),len(lon)],dtype='f')
    filtData[0:obsData.shape[0],:]=obsData

    #### # Filter (KELVIN)
    #################################################
    print("\n##############################\nFiltering %s\n"%waveName)

    wave = filtData
    wave_longname="Kelvin Waves in "+filtVar.upper()
    wave_filter="Straub & Kiladis (2002) to 20 days"
    wave_wavenumber=waves[waveName][0]
    wave_period=waves[waveName][1]
    wave_depth=waves[waveName][2]
    wave_units=unit

    wave[:,:]=kf_filter(filtData[:,:],obsPerDay,\
                   wave_period[0],wave_period[1],\
                   wave_wavenumber[0],wave_wavenumber[1],\
                   wave_depth[0],wave_depth[1],waveName)
    waveData = xr.DataArray(wave[0:len(obsTime),:].squeeze(),coords=[obsTime,lon],dims=['time','lon'])

    #differentiate twice
    waveDatadt = waveData.differentiate('time')
    waveDatadt2 = waveDatadt.differentiate('time')

    #retain only strong waves and where d/dt(d/dt(wave)) < 0 [a maximum]
    #further restrict only to Pacific ocean
    print('considering 1sigma waves!')
    waveCrests = waveDatadt.where( (waveData/waveData.std(dim=('time'))>1.0) & (xr.ufuncs.fabs(waveDatadt) <= 1.0e-14) &
                                      (waveDatadt2 < 0) & (waveData.lon >= 0.0) &
                                      (waveData.lon <= 360.0))

    time_in_days = np.arange(0,len(waveData.time)*dt,dt)/24
    [X,Y] = np.meshgrid(waveData.lon,time_in_days)
    plt.contourf(X,Y,waveData)
    plt.colorbar()

    cs = plt.contour(X,Y,waveCrests,colors='k',levels=[0])

    plt.close()

    lon_waves = []
    time_waves = []
    for segs in cs.allsegs[0][:]:
        for i in np.arange(0,len(segs[:,0])):
            lon_waves.append(segs[i,0])
            time_waves.append(obsTime.sel(time=obsTime[0].squeeze()+pd.to_timedelta(segs[i,1],unit='d'),method='nearest') )

    del(filtData)
    del(obsData)

    return lon_waves, time_waves, waveData, waveDatadt
