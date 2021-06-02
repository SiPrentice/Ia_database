#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:25:05 2021

@author: sprentice
"""

import requests
import pandas as pd
import json
import os
from collections import OrderedDict
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
import datetime
import numpy as np
from scipy.interpolate import UnivariateSpline as spline
import matplotlib.pyplot as plt

# Some user input here
downloadRecent = True # set as True to use the Fritz API to get recent objects
main_csv = '2021_SNeIa.csv'
csv = 'tableDownload.csv'
save_csv = True #overwrites the main csv, normally set to True
maxRedshift = 0.025 # sets the redshift limit for frtching recent objects
no_older_than = 35 # the limit for the discovery date in days since now 
####

gsheet = pd.read_csv(f'./{main_csv}', delimiter ='\t')
tokens = np.loadtxt('./tokens.txt', unpack = True, dtype='str')


# API related 
# Fritz API token
token = tokens[0]



# check if the plots directory exists
if os.path.isdir('./plots') == False:
    print('Making ./plots/')
    os.mkdir('./plots')



# TNs requirements

TNS="www.wis-tns.org"
#TNS="sandbox.wis-tns.org"
url_tns_api="https://"+TNS+"/api/get"

# API key for Bot
api_key = tokens[1]
YOUR_BOT_ID = tokens[2]
YOUR_BOT_NAME = tokens[3]

# list that represents json file for search obj
search_obj=[("ra",""), ("dec",""), ("radius",""), ("units",""), ("objname",""), 
            ("objname_exact_match",0), ("internal_name",""), 
            ("internal_name_exact_match",0), ("objid",""), ("public_timestamp","")]
# list that represents json file for get obj
get_obj=[("objname",""), ("objid",""), ("photometry","0"), ("spectra","1")]

# current working directory
cwd=os.getcwd()
# directory for downloaded files
download_dir=os.path.join(cwd,'downloaded_files')


# Columns of the google sheet
cols = ['Name',
 'ztf name',
 'Obj. Type',
 'ra/dec',
 'Redshift',
 'Max spec',
 'PESSTO obj?',
 'Sources',
 'Comments',
 'Disc. Internal Name',
 'Discovery Date (UT)',
 ]


# define the csv from the google sheet


# Functions

# TNS related functions

def format_to_json(source):
    # change data to json format and return
    parsed=json.loads(source,object_pairs_hook=OrderedDict)
    result=json.dumps(parsed,indent=4)
    return result

# function for search obj
def search(url,json_list):
  try:
    # url for search obj
    search_url=url+'/search'
    # headers
    headers={'User-Agent':'tns_marker{"tns_id":'+str(YOUR_BOT_ID)+', "type":"bot",'\
             ' "name":"'+YOUR_BOT_NAME+'"}'}
    # change json_list to json format
    json_file=OrderedDict(json_list)
    # construct a dictionary of api key data and search obj data
    search_data={'api_key':api_key, 'data':json.dumps(json_file)}
    # search obj using request module
    response=requests.post(search_url, headers=headers, data=search_data)
    # return response
    return response
  except Exception as e:
    return [None,'Error message : \n'+str(e)]

# function for get obj
def get(url,json_list):
  try:
    # url for get obj
    get_url=url+'/object'
    # headers
    headers={'User-Agent':'tns_marker{"tns_id":'+str(YOUR_BOT_ID)+', "type":"bot",'\
             ' "name":"'+YOUR_BOT_NAME+'"}'}
    # change json_list to json format
    json_file=OrderedDict(json_list)
    # construct a dictionary of api key data and get obj data
    get_data={'api_key':api_key, 'data':json.dumps(json_file)}
    # get obj using request module
    response=requests.post(get_url, headers=headers, data=get_data)
    # return response
    return response
  except Exception as e:
    return [None,'Error message : \n'+str(e)]

# function for downloading file
def get_file(url):
  try:
    # take filename
    filename=os.path.basename(url)
    # headers
    headers={'User-Agent':'tns_marker{"tns_id":'+str(YOUR_BOT_ID)+', "type":"bot",'\
             ' "name":"'+YOUR_BOT_NAME+'"}'}
    # downloading file using request module
    response=requests.post(url, data={'api_key':api_key}, headers=headers, stream=True)
    # saving file
    path=os.path.join(download_dir,filename)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in response:
                f.write(chunk)
        print ('File : '+filename+' is successfully downloaded.')
    else:
        print ('File : '+filename+' was not downloaded.')
        print ('Please check what went wrong.')
  except Exception as e:
    print ('Error message : \n'+str(e))
    

    
# Fritz API function    

def api_meta(method, endpoint, data=None):
    headers = {'Authorization': f'token {token}'}
    response = requests.request(method, endpoint, json=data, headers=headers)
    return response



# My functions for other processes

def get_TNS_name(ztf):    
    
    search_obj=[("ra",""), ("dec",""), ("radius","5"), ("units","arcsec"), 
            ("objname",""), ("objname_exact_match",0), ("internal_name", ztf), 
            ("internal_name_exact_match",0), ("objid",""), ("public_timestamp","")]     

    response=search(url_tns_api,search_obj)

    if None not in response:
        a = response.json()
        return a["data"]['reply'][0]['objname']
    else:
        return 'None'    


def get_coords(ra,dec):
    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    string = c.to_string('hmsdms')
    string = string.replace('h',':')
    string = string.replace('m', ':')
    string = string.replace('d', ':')
    string = string.replace('s', '')
    
    return string
    


### Functions for using Fritz API to get recent objects
### saves a new csv file as tableDownload.csv

def api(method, endpoint, data = None):
    
    headers = {'Authorization': f'token {token}'} 
    
    response = requests.request(method, endpoint, json=data, headers=headers)
    
    #print(response.status_code)
    if 400 == response.status_code:
        print(f'HTTP code: {response.status_code}, {response.reason}')
        print(f'Endpoint requested: {endpoint}')
    
    return response 

def get_recent_objects(maxRedshift=maxRedshift, no_older_than=no_older_than):
    '''
    Uses the Fritz API to fetch SNe Ia saved earlier than 35 days from the time now with limiting redshift set to
    maxRedshift.
    Output is a dictionary called 'data'.
    '''
    time_now = Time(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')).mjd
    savedAfter = Time(time_now - no_older_than, format = 'mjd').isot
    print(f'Getting objects saved after {savedAfter}')
    numPerPage = 100
    pageNumber = 1
    group_ids = '41'
    classifications = 'Sitewide Taxonomy: Ia, Sitewide Taxonomy: Ia-02cx, Sitewide Taxonomy: Ia-91T,\
    Sitewide Taxonomy: Ia-03fg, Sitewide Taxonomy: Ia-18byg, Sitewide Taxonomy: Ia-91bg, Sitewide Taxonomy: Ia-norm\
    Sitewide Taxonomy: Ia-pec'
    
    # construct the URL      
#    endpoint = f'https://fritz.science/api/sources?savedAfter={savedAfter}&numPerPage={numPerPage}&\
#    pageNumber={pageNumber}&classifications={classifications}&maxRedshift={maxRedshift}'
    
    # construct the URL WHILE MAXREDSHIFT IS BROKEN     
#    endpoint = f'https://fritz.science/api/sources?savedAfter={savedAfter}\
#    &classifications={classifications}&group_ids={group_ids}'
    
    endpoint = f'https://fritz.science/api/sources?savedAfter={savedAfter}\
    &classifications={classifications}&maxRedshift={maxRedshift}&group_ids={group_ids}'

    sources = api('GET', endpoint, data = None)    
    a = sources.json()
    data = a['data']['sources']
    
    return data

tableDownload_headers = ['Source ID',
                         'Alias',
                         'RA (deg)',
                         'Dec (deg)',
                         'RA (hh:mm:ss)',
                         'Dec (dd:mm:ss)',
                         'Redshift',
                         'Classification',
                         'Groups',
                         'Date Saved',
                         'Finder']    


def generate_row(source, maxRedshift=maxRedshift):
    '''Takes the source row for the Fritz API output and returns a new row to add to the csv'''
    
    tmp = []
    
    tmp.append( source.get('id', '') )
    tmp.append('')
    for i in range(2):
        tmp.append( source.get('ra', '') )
        tmp.append( source.get('dec', '') )
    tmp.append(source.get('redshift', ''))
    for j in range(4):
        tmp.append('')
        
    return tmp   
 

def make_csv(data, save_new_csv = True):
    '''Makes the tableDownload.csv file used by the Ia pipeline. The headers are imported from above and the
    csv rows filled from the 'data' file returned by the api.
    '''

    df = [generate_row(source) for source in data if source.get('redshift', 3) <= maxRedshift]
        
    # set the dataframe with the required headers
    csv = pd.DataFrame(data = df, columns = tableDownload_headers)
    
    if save_new_csv:
        print(f'Saving ./tableDownload.csv\n')
        csv.to_csv('./tableDownload.csv', index = False)
    else:    
        return csv
    
#################################

# make a dataframe from the ztf csv
def ztf_dataframe(csv):
    df = pd.read_csv(csv)
    
    df['TNS name'] = [get_TNS_name(sn) for sn in df['Source ID']]
    
    columns_to_use =  ['Source ID','TNS name', 'RA (hh:mm:ss)',
       'Dec (dd:mm:ss)', 'Redshift', 'Classification', 'Date Saved',
       ]

    df = df[columns_to_use]
        
    df['ra/dec'] = [get_coords(df['RA (hh:mm:ss)'][j], df['Dec (dd:mm:ss)'][j]) for j in range(len(df['Classification'])) ]  
    
    #print(df.head())
    
    return df

def get_created_date(sn):
    response = api_meta('GET', f'https://fritz.science/api/candidates/{sn}')
    
    print(f'HTTP code: {response.status_code}, {response.reason}')
    
    a = response.json()
    date = a['data'].get('created_at', None)

    print(f'{sn} created at {date}')
        
    return date


def create_csv_entry(df):
    
    df['Name'] = df['TNS name']
    df['ztf name'] = df['Source ID']
    df['Disc. Internal Name'] = df['Source ID']
    df['Obj. Type'] = df['Classification']
    
    # get the date the entry was created
    disc_dates = []
    for zname in df['Source ID']: 
        print(f'\nGetting created date for {zname} using Fritz API...')
        date = get_created_date(zname)
        disc_dates.append(date)
    df['Discovery Date (UT)']= disc_dates
    
    for header in cols:
        if header not in list(df.columns):
            df[header] = ['' for j in df.Name]
            
    df = df[cols]
    
    return df
    
    
def new_object(row, main = gsheet):
    '''sn is the name of the transient, main is the reference dataframe'''
    
    sn_list = list(main.Name.values)
    ztf_list = list(main['ztf name'].values)
    
    if row.Name in sn_list:
        return False
    
    elif row['ztf name'] in ztf_list:
        return False
    
    else:
        return True
    

def drop_rows(df): 
    drop_list =[]
    print('\n')
    for i in range(len(df)):
        row = df.iloc[i]

        if new_object(row):
            print(f'{row.Name} is new')
        
        else:
            zname = row['ztf name']
            print(f'Dropping {row.Name}/{zname}, it already exists in {main_csv}')
            drop_list.append(i)
            
    df = df.drop(drop_list).reset_index(drop = True)
    return df

    
def process_ztf_csv(csv, save = True):    
    print(f'*** Processing {csv} ***')
    df_ztf = ztf_dataframe(f'./{csv}')
    
    
    print(f'\nCreating data frame for entry into {main_csv}')
    df_ztf = create_csv_entry(df_ztf)
    
    df_ztf = drop_rows(df_ztf)
    
    if df_ztf.shape[0] > 0:
        df = pd.concat([gsheet, df_ztf], ignore_index=True, sort=False)

        print(f'\nSaving new version of {main_csv}')
        
        if save:
            df.to_csv(f'./{main_csv}', sep = '\t', index = False)
        return df
        
    else:
        print(f'\nNo new objects to include, {main_csv} not updated.')
    
        return gsheet  


# The functions are for obtaining the photometry of an object

def api_phot(method, endpoint, data=None, obj_id =''):
    headers = {'Authorization': f'token {token}'}
    response = requests.request(method, endpoint + obj_id + '/photometry', json=data, headers=headers)
    return response

def get_photometry(json, band = 'ztfr'):
    
    data = json['data']
    
    mjd, m = [], []
    
    for obs in data:
        if obs.get('filter', '') == band:
            
            mjd.append( obs.get('mjd', float('nan') ) )
            m.append( obs.get('mag', float('nan') ))
            
    return np.array(mjd), np.array(m)      


# These functions are for getting the spectra dates
def api_spectra(method, endpoint, data=None):
    headers = {'Authorization': f'token {token}'}
    response = requests.request(method, endpoint, json=data, headers=headers)
    return response

def get_spectra_dates(zname):
    
    print('Fetching the spectra dates')
    response  = api_spectra('GET', f'https://fritz.science/api/sources/{zname}/spectra')
    print(f'HTTP code: {response.status_code}, {response.reason}')
    
    a = response.json()
    
    spectra = a['data']['spectra']
    
    mjd_list = []
    
    for s in spectra:
        observed_at_isot = s['observed_at']
        mjd = Time(observed_at_isot).mjd
        mjd_list.append(mjd)
    
    return mjd_list


# This function fits the light curve and produces the output plots

def fit_LC( t_ref , m_ref , runs = 100, tmax_only = True, band = 'ztfr', name ='', mjd_now = 0):
    
    
    # Normalise the reference LC. The first two may be needed in order to plot some relative time
    t_last = max(t_ref)
    t_first = min(t_ref)
    t_ref = t_ref - min(t_ref) 
    m_ref = m_ref - min(m_ref)
    
    
    # load in the template LC
    t, m = np.loadtxt(f'./templates/template_for_{band}.txt', unpack = True)
    
    # set the offsets
    m_off = 1
    t_off = 1
    
    
    results = [ [1e9, t_off, m_off]   ]
    final_t = []
    final_m = []
    
    for j in range(runs):
        # get the prvious values
        t_off = results[-1][1]
        m_off = results[-1][2]
        
        # make small random variations to them
        t_off = np.random.normal(t_off, max(t_off * 0.5, 1) )
        # Constraints, otherwise the code may not converge
        while 20 > t_off < -10:
            t_off = np.random.normal(t_off, max(t_off * 0.5, 1) )

        m_off = np.random.normal(m_off, max(m_off * 0.5, 0.2) )
        while 5 > m_off < -5:
            m_off = np.random.normal(m_off, max(m_off * 0.5, 0.2) )
        
        
        # set the new position for the template LC in mag and t space
        t_new = t - t_off
        m_new = m - m_off
        
        # fit a spline
        fit = spline(t_new, m_new, k = 1, s =0)
                
        
        # evaluate the spline at the point of the reference LC
        m_fit = fit(t_ref)
        
        mae = sum( abs( np.array(m_ref) - m_fit )**2 ) / len(m_ref)
    
        if mae < results[-1][0]:
            results.append( [mae, t_off, m_off]        )
            final_t = t_new
            final_m = m_new
    
    tmax = (t[np.argmin(m)] - results[-1][1]) - t_ref[-1] 
    
    
    # for this object get the spectra mjds from fritz
    # first, get the spectra dates in the Fritz
    spectra_mjds = get_spectra_dates(name)

    # plot the LCs
    plt.plot(final_t, final_m, color = 'grey', linestyle = 'dotted', label ='Template')
    plt.scatter(t_ref, m_ref, marker = 'o', label = band, zorder = 0)    
    plt.axvline(x = mjd_now - t_first, color = 'tab:orange', lw = 1, label = 'Time of plot')
    
    for s in spectra_mjds:
        plt.axvline(x = s - t_first, linestyle = 'dashed', color='lightgrey', lw = 1, zorder =0)
        plt.text(s - t_first, -0.1, 'S', ha = 'center', fontsize = 8)
    
    plt.legend()
    plt.gca().invert_yaxis()
    plt.minorticks_on()
    plt.xlabel('Time since first observation', fontsize = 12)
    plt.ylabel('Magnitude + offset', fontsize = 12)
    plt.savefig(f'./plots/{name}_{band}.pdf', bbox_inches = 'tight')
    plt.close()  
    
    if tmax_only:
        return tmax, t_last + tmax
    else:
        return  tmax, (final_t, final_m), np.array(results)

    
def clean_LC(t_ref,m_ref):
    # some cleaning, must do t_ref before m_ref
    m_ref = np.array(m_ref, dtype='float')
    t_ref = t_ref[~np.isnan(np.array(m_ref))]
    m_ref = m_ref[~np.isnan(np.array(m_ref))]
    
    return np.array(t_ref), np.array(m_ref)

    
def get_tmax(sn, time_now):
    print(f'\nFetching photometry for {sn}')
    response  = api_phot('GET', 'https://fritz.science/api/sources/', obj_id = sn)
    print(f'HTTP code: {response.status_code}, {response.reason}')

    a = response.json()
    
    # creat a list of lists to hold the time to max and the mjd of max
    tmax_list = [[],[]]
    for band in ('ztfg', 'ztfr'):
        
        # get the photometry
        t, m = get_photometry(a, band = band)
        
        # remove None values
        t, m = clean_LC(t, m)
        
        # fit for maximum, rejecting if there is not enough photometry
        if len(m) > 1:
            tmax = fit_LC(t,m, band = band, name = sn, mjd_now = time_now)
        
            isot_max = Time(tmax[1], format = 'mjd').isot
            print(f'Predicted {band} maximum on {isot_max}')
            tmax_list[0].append(tmax[0])
            tmax_list[1].append(tmax[1])
    
    
    return ( np.mean(tmax_list[0]), np.mean(tmax_list[1]) )



def set_for_obs(df, obs_file = './OBSERVE_SN.txt'):
    print(f'\n*** Setting {obs_file} for observations ***')
    
    # set a dictionary of the status for each sn
    sn_dict = {}
    
    # check if the obsfile exists
    if os.path.isfile(obs_file):
        rows = np.loadtxt(obs_file, dtype = 'str', delimiter = '\t')
    
        for row in rows:
            sn_dict[row[0]] = [*row[1:]]
        
    # get the time now
    time_now = Time(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')).mjd
    
    # now iterate through the sne in the main csv
    main = []
    
    for i in range(df.shape[0]):
        zname = df['ztf name'][i]
        coordinates = df['ra/dec'][i]
        z = df['Redshift'][i]
        has_max = df['Max spec'][i]
        
        
        #print(df['Discovery Date (UT)'][i])
        if str(df['Discovery Date (UT)'][i]) != 'nan':
            
            # set the time of discovery in ztf, don't want to process old objects
            t0 = Time(df['Discovery Date (UT)'][i]).mjd
            
            # if the object exists in the dictionary, take that, if not create a new row
            sn_status = sn_dict.get(zname, [z, coordinates, None, '-'])
                
            
            # only take new objects
            if time_now - t0 <35:               
                
                # now get the time since max and plot the light curves
                tsince, mjd_max = get_tmax(zname, time_now)
                isot_max = Time(mjd_max, format = 'mjd').isot
                
                
                # if the SN has not been added, we want to include the isot_max
                if sn_status[2] == None:
                    sn_status[2] = isot_max
                
                # Now check each object. Set 'complete' those with max spectrum. pandas gives blank elements as nan
                if str(has_max) == 'y':
                    print(f'{zname} has maximum light spectrum.')
                    sn_status[3] = 'complete'
                    main.append([zname, *sn_status ])
                
                # Make those with no max spectrum but tmax in the future active
                elif ((time_now - mjd_max) < 10):
                    print(f'{zname} is active. Please update observing schedule.')
                    sn_status[3] = '*** active ***'
                    
                    main.append([zname, *sn_status ])
                
                # deactivate any without max spectrum and no max spectrum
                else:
                    print(f'{zname}, peak passed {(time_now - mjd_max):.1f} days ago.')

                    sn_status[3] = '-'
                    main.append([zname, *sn_status ])
    
    print(f'\nSaving {obs_file}')
    np.savetxt(obs_file, main, fmt="%s", delimiter = '\t', header = 'ztf name\tz\tra/dec\tisot of max\tstatus')
    
    print(f'Pipeline complete\n')
    
    
## Run the pipeline
if downloadRecent:
    data = get_recent_objects()
    make_csv(data)
    
df = process_ztf_csv(csv, save = save_csv)
set_for_obs(df)    