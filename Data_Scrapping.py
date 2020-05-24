from bs4 import BeautifulSoup
import requests

def url_generator(state,page_number):
    url = []
    j = 2
    for i in range(len(state)):
        url.append("https://timesofindia.indiatimes.com/india/" + state[i])
        for j in range(2,page_number+1):
            url.append("https://timesofindia.indiatimes.com/india/" + state[i] + "/" + str(j))
    return url

def news_scrapper(url):
    News_Headline = []
    for i in range(len(url)):
        print(url[i])
        page = requests.get(url[i])
        soup = BeautifulSoup(page.content, 'html.parser')
        news = soup.find(id="c_articlelist_stories_2").get_text()
        scrapped_headline = news.splitlines()
        scrapped_headline = [x for x in scrapped_headline if x]
        del scrapped_headline[-1]
        News_Headline = News_Headline + scrapped_headline
    return News_Headline

def state_wise_news(all_news):
    State_News = []
    temp_list = []
    for i in range(len(all_news)):
        temp_list.append(all_news[i])
        if len(temp_list) == 44:
            State_News.append(temp_list)
            temp_list=[]
    return State_News

def find_state_news(Indian_States,state_name,state_news):
    index = Indian_States.index(state_name)
    return state_news[index]
    














        
    


