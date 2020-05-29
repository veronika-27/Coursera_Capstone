#!/usr/bin/env python
# coding: utf-8

# ## Problem Description and Background
# 
# <font>According the Canadian statistics (https://www12.statcan.gc.ca/census-recensement/2011/as-sa/fogs-spg/Facts-csd-eng.cfm?Lang=eng&GK=CSD&GC=3520005) Toronto has an estimated population of over 2.8 million in 2016, which makes it the 4th most populous city in North America and the most populous city in Canada. It is also the largest urban and metro area, with a population density of 4,149.5 people per square kilometer. According to the 2006 Census, foreign-born people have been 45.7 % of the population of Toronto (https://www12.statcan.gc.ca/english/census06/analysis/immcit/charts/chart4.htm), which convert it in the second-highest percentage of foreign-born residents of all world cities after Miami. The data related to the population and its diversity made me believe that Toronto attracts a lot a people from all over the world for business, tourism, etc. and the property business and the investment properties can be pretty rewarding if the right choice is made.</font>
# 
# The first step and one of the key factors that has to be considered when shopping for an income property is the review of the neighborhood's livability and facilities. After that few key factors has to be considered as well: the neighborhood vacancy rate, the local selling prices, and the average rent in the area in order to determine the financial feasibility. 
# 
# In order to find the best place for investment properties, an analysis of the neighborhoods in Toronto will be made. The primary goal of this project will be to evaluate and determine which neighborhood/s would be most appropriate for investment in property. As they are a lot of types of potential tenants and the neighborhood in which the property would be bought will determine the types of tenants the perspective of a family/couple as a potential tenants will be taken (based on the assumptions that families or couples are generally better tenants than singles because they are more likely to be financially stable and pay the rent regularly).
# 
# Another assumption taken into consideration is that the home values can be affected by the 'energy' of the neighborhood in which they are located. And form the family/couple perspective the affects will be limited to: more green spaces, parks, restaurants, theaters; less breaks and enters, less nightclub and bars, more child care spaces, the financial status of the neighbors (debt risk score and income after taxes), total business establishments, total local jobs. Also, the home prices by neighborhoods will be revised, in order to help the potential investors (buyers).
# 
# This analysis is meant to be helpful for potential property investors or families trying to find the right neighborhood in Toronto, where to move/live. According to one of the newest real estate forecasts (This analysis is meant to be helpful for potential property investors or families trying to find the right neighborhood in Toronto, where to move/live. According to one of the newest real estate forecasts 'prices are still trending upward, but Coronavirus containment efforts pull prices down. It is likely that prices will be lower in 2021.'(05.05.2020; https://www.mortgagesandbox.com/toronto-real-estate-forecast), so probably more people will take advantage of this report. 
# 
# 
# ## Data Description
# 
# The data that will be used to make a 'profile' of the neighborhoods is the data used during the assignment form week 3 (the Foursquare location data will be used in order to find the neighborhoods with less nightclub and bars, white more parks, restaurants, theaters etc.) combined with more data from this Canadian open data portal (https://open.toronto.ca/catalogue/?search=neighbourhood&sort=score%20desc). The information related to child care spaces, debt risk score, household income after taxes, total business establishments, total local jobs, home prices by neighborhoods will be extracted from the open data portal and properly handled, arranged and converted into pandas data frame and combined with the data form Foursquare in order to help me find the 'right' neighborhood or group of neighborhoods for rental property form family perspective.
# 

# In[ ]:




