import requests
from bs4 import BeautifulSoup

url = 'https://scholar.google.fr/scholar?start=0&hl=en&as_sdt=2005&sciodt=0,5&cites=18446710290766215629&scipsc='

re = requests.get(url)
soup = BeautifulSoup(re.content, 'html5lib')

papers = soup.find_all('div', attrs={'class':'gs_r gs_or gs_scl'})

for paper in papers:
    print(paper.a['href'])