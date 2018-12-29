import urllib.request, json


class CrawlData:
    def __init__(self):
        self.API_Key = "9fe303a1e043be763beb0db7c5128b8b"
        self.Shared_secret = "fbe0b9c7fa4914d1e8334050589cd142"
        self.prefix = "http://ws.audioscrobbler.com/2.0/?method=artist.getinfo&artist="
        self.postfix = "&format=json"

    def make_string(self, artist_name="Cher"):
        return self.prefix + artist_name + "&api_key=" + self.API_Key + self.postfix


crawl = CrawlData()
print(crawl.make_string())
with urllib.request.urlopen(crawl.make_string()) as url:
    data = json.loads(url.read().decode())
    print(data)






