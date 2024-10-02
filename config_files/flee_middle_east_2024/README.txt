------------------------------------------------------------------------------------------------------
1. Locations.csv

Plan: 
Because Lebanon is small and we have limited time, I suggest we make a location graph on the district level, 
and have 3 locations abroad in Syria: Damascus, Homs and Tartus would make most sense initially. 
I think UNHCR and other organisations are relatively unlikely to establish camps in Syria, given the regime and situation there.
As for Lebanon, its 9 districts are shown here: https://en.wikipedia.org/wiki/Districts_of_Lebanon
Probably, we want to use the largest settlement in each district as centre point, and then interconnect those along with the 3 locations abroad.
For obvious reasons, I don't think people will Flee into Israel.
This is a location graph with 12 locations. 

Town Implementation: 
Use the 9 governorates of Lebanon: https://en.wikipedia.org/wiki/Governorates_of_Lebanon
- We have used the capital city as the location name and the populaton of the district.
- These are taken from citypopulation.de - which is currently down. 
Use 3 towns in Syria: Damascus, Homs and Tartus 
- Chosen based on proximity to Lebanon
- Info taken from the table here: https://en.wikipedia.org/wiki/List_of_cities_in_Syria
- All sources are citypopulation.de 
- Use the 3 city names, use governorates and latitude/longitude of city, with city population. 

Conflict Implementation:

Camp Implementation:

------------------------------------------------------------------------------------------------------

2. Routes.csv

Lebanon: 
Connected all neighbouring governorates to each other. 
Then took the distance corresponding to the shortest journey time in google maps between the captial cities of the Governorates. 
All distances in km. 

Syria: 
Connected to nearest governorates as the crow flies. 
Used google maps, shortest journey time distance between two cities. 
All distances in km. 


------------------------------------------------------------------------------------------------------
3. closures.csv 

Left blank for now. 

------------------------------------------------------------------------------------------------------
4. sim_period.csv

September 1st 2024 for 32 days at the moment. 

------------------------------------------------------------------------------------------------------
5. registration_corrections.csv

Left blank for now.