'''
1) 구글 뉴스에서 특정 검색어 입력시 50개의 URL 주소, 기사 제목, 등록 시간, 기사 내용

'''
import requests
from bs4 import BeautifulSoup

url = 'https://www.google.com/search?biw=811&bih=576&tbm=nws'
html = requests.get(url).text.strip()
soup = BeautifulSoup(html, 'html5lib')
links = soup.find_all('a')
for link in links:
    print(link.get('href'))




def google_search(keyword):
    pass
    # 구글 news 검색 url
    #url = 'https://www.google.com/search?biw=811&bih=576&tbm=nws'
    # for page in range(5):
    #     print(f'===page {page}===')
    #     req_parms = {
    #         'lst-ib': keyword,
    #         'csb ch': page
    #     }
    #     response = requests.get(url, params=req_parms)
    #     html = response.text.strip()
    #     soup = BeautifulSoup(html, 'html5lib')
    #     results = soup.select('div h3 a.l lLrAF')
    #     for link in results:
    #         news_url = link.get('href')
    #         news_title = link.text
    #         print(news_url, news_title)



if __name__ == '__main__':
    google_search('중국')