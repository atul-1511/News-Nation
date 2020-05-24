import sys 
import os
sys.path.append(os.path.abspath(r"C:\Users\Atul Kumar/News/"))
from Data_Scrapping import url_generator,news_scrapper,state_wise_news,find_state_news


Indian_States = ["andhra-pradesh","arunachal-pradesh","assam","bihar","chhattisgarh","goa",
                 "gujarat","haryana","himachal-pradesh","jharkhand","karnataka","kerala",
                 "madhya-pradesh","maharashtra","manipur","meghalaya","mizoram","nagaland",
                 "orissa","punjab","rajasthan","sikkim","tamil-nadu","telangana","tripura",
                 "uttar-pradesh","uttarakhand","west-bengal"]

page_number = 1

url = url_generator(Indian_States,page_number)
scrapped_headlines = news_scrapper(url)
state_news = state_wise_news(scrapped_headlines)

query = "mizoram"
News = find_state_news(Indian_States,query,state_news)

