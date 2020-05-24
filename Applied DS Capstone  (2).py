#!/usr/bin/env python
# coding: utf-8

# # Segmenting and Clustering Neighborhoods in Toronto 

# # Part 1 - building the dataframe of the postal codes 

# In[1]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab

get_ipython().system('pip install geopy')
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

get_ipython().system("conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab")

import folium # map rendering library

print('Libraries imported.')


# ## Scraping the data from Wikipedia

# In[2]:


# import the library we use to open URLs
import urllib.request

# specify which URL/web page we are going to be scraping
url = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"

# open the url using urllib.request and put the HTML into the page variable
page = urllib.request.urlopen(url)

# import the BeautifulSoup library so we can parse HTML and XML documents
from bs4 import BeautifulSoup

# parse the HTML from our URL into the BeautifulSoup parse tree format
soup = BeautifulSoup(page, "lxml")

print(soup.prettify())


# In[15]:


PC_table = soup.find('table')
Table_fields = PC_table.find_all('td')

PostalCode=[]
Borough=[]
Neighborhood=[]

for i in range(0, len(fields), 3):
    PostalCode.append(fields[i].text.strip())
    Borough.append(fields[i+1].text.strip())
    Neighborhood.append(fields[i+2].text.strip())
        
df_PC = pd.DataFrame(data=[PostalCode, Borough, Neighborhood]).transpose()
df_PC.columns = ['PostalCode', 'Borough', 'Neighborhood']
df_PC.head()


# ### Ignoring (dropping) cells with a borough that is Not assigned

# In[18]:


df_PC['Borough'].replace('Not assigned', np.nan, inplace=True)
df_PC.dropna(subset=['Borough'], inplace=True)
df_PC.head(5)


# ### Combining the neighborhoods with same postal codes 

# In[19]:


df_PC_new = df_PC.groupby(['PostalCode', 'Borough'])['Neighborhood'].apply(', '.join).reset_index()
df_PC_new.columns = ['PostalCode', 'Borough', 'Neighborhood']
df_PC_new


# ### Cheking for "Not assigned' neighborhood

# In[27]:


df_PC_new['Neighborhood'].where(df_PC_new['Neighborhood']=='Not assigned')


# #### There is not any cell where the neighborhood is "Not assigned", so no need to replace it with the borough. 

# In[29]:


df_PC_new.shape


# # Part 2 - Adding the latitude and the longitude coordinates, in order to use Foursquare

# In[62]:


df_latlong = pd.read_csv('http://cocl.us/Geospatial_data')
df_latlong


# In[66]:


df_latlong.columns = ['PostalCode', 'Latitude', 'Longitude']
df_latlong


# In[67]:


df_new = pd.merge(df_PC_new, df_latlong, on=['PostalCode'], how='inner')
df_new


# # Part 3- Exploring and clustering the neighborhoods in Toronto

# In[68]:


address = 'Toronto, Canada'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))


# #### Creating a map of Toronto

# In[69]:


#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

print('Library imported.')


# In[73]:


# create map of Toronto using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(df_new['Latitude'], df_new['Longitude'], df_new['Borough'], df_new['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='red',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# ## Exploring the neighborhoods in Toronto 

# #### Defining Foursquare Credentials and Version

# In[154]:


CLIENT_ID = '***' # y31GBCFS3Y421BJBDKSA52QMGZYL15XYF4TCGTO25V2GHSSKTour Foursquare ID
CLIENT_SECRET = '***' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[99]:


df_newt = df_new[df_new['Borough'].str.contains('Toronto')]

toronto_data = df_newt.reset_index(drop=True)
toronto_data


# ### Map only of the boroughs containing "Toronto" 

# In[100]:


map_newt = folium.Map(location=[latitude, longitude], zoom_start=12)

for lat, lng, borough, neighborhood in zip(toronto_data['Latitude'], toronto_data['Longitude'], toronto_data['Borough'], toronto_data['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=3,
        popup=label,
        color='red',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.5,
        parse_html=False).add_to(map_newt)  
    
map_newt


# ####  I have decided to explore the area near Qeen's Park. 
# 
# ##### I saw that the part near The Royal Ontario Museum is one of the busiest in the city and I also, wanted to check the administrative area. 

# In[103]:


toronto_data.loc[37, 'Neighborhood']


# In[104]:


neighborhood_latitude = toronto_data.loc[37, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = toronto_data.loc[37, 'Longitude'] # neighborhood longitude value

neighborhood_name = toronto_data.loc[37, 'Neighborhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


# #### Now, let's get the top 100 venues that are near Qeen's Park within a radius of 500 meters.

# In[155]:


LIMIT= 100
radius= 500

url= url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
    
url


# In[106]:


results = requests.get(url).json()
results


# In[107]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# #### Cleaning the json and structure it into a *pandas* dataframe

# In[109]:


venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues


# In[110]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# ## Exploring Neighborhoods in Toronto

# #### Creating a function to repeat the same process to all the neighborhoods in Toronto 

# In[120]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[122]:


Toronto_venues = getNearbyVenues(names=toronto_data['Neighborhood'],
                                   latitudes=toronto_data['Latitude'],
                                   longitudes=toronto_data['Longitude']
                                  )


# In[123]:


print(Toronto_venues.shape)
Toronto_venues.head()


# In[124]:


Toronto_venues.groupby('Neighborhood').count()


# ####  Finding out how many unique categories can be curated from all the returned venues

# In[126]:


print('There are {} uniques categories.'.format(len(Toronto_venues['Venue Category'].unique())))


# ## Analyzing Each Neighborhood

# In[127]:


# one hot encoding
Toronto_onehot = pd.get_dummies(Toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
Toronto_onehot['Neighborhood'] = Toronto_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [Toronto_onehot.columns[-1]] + list(Toronto_onehot.columns[:-1])
Toronto_onehot = Toronto_onehot[fixed_columns]

Toronto_onehot.head()


# In[128]:


Toronto_onehot.shape


# #### Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category

# In[129]:


Toronto_grouped = Toronto_onehot.groupby('Neighborhood').mean().reset_index()
Toronto_grouped


# #### Let's confirm the new size

# In[130]:


Toronto_grouped.shape


# #### Let's print each neighborhood along with the top 5 most common venues

# In[132]:


num_top_venues = 5

for hood in Toronto_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = Toronto_grouped[Toronto_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# #### Let's put that into a *pandas* dataframe

# ##### First, let's write a function to sort the venues in descending order.

# In[133]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# ##### Now let's create the new dataframe and display the top 10 venues for each neighborhood.

# In[150]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = Toronto_grouped['Neighborhood']

for ind in np.arange(Toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(Toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# ## Clustering Neighborhoods

# In[151]:


# set number of clusters
kclusters = 7

Toronto_grouped_clustering = Toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(Toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:15] 


# ##### Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.

# In[152]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

Toronto_merged = toronto_data

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
Toronto_merged = Toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

Toronto_merged.head() # check the last columns!


# ##### Finally, let's visualize the resulting clusters

# In[153]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(Toronto_merged['Latitude'], Toronto_merged['Longitude'], Toronto_merged['Neighborhood'], Toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[ ]:




