import numpy as np
import requests
from bs4 import BeautifulSoup
import json
import locale
import datetime
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from DeepReg import *


def main():

    URL = 'https://www.nadlan.gov.il/Nadlan.REST/Main/GetDataByQuery'
    # params = {'query': 'שמעון בן שטח 5,  תל אביב -יפו'}
    params = {'query': 'רחוב ברנשטין כהן, תל אביב -יפו '}
    # params = {'query': 'שכונת צפון יפו, תל אביב -יפו'}
    spesific = 'SpecificAddressData'  # 'AllResults'

    headers = {"User-Agent": 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:68.0) Gecko/20100101 Firefox/68.0'}
    data = requests.get(URL, headers=headers, params=params)

    json_data = data.json()
    # json_data['PageNo'] = 1
    json_data['PageNo'] = 1

    URL = 'https://www.nadlan.gov.il/Nadlan.REST/Main/GetAssestAndDeals'
    r = requests.post(URL,
                      headers={"User-Agent":
                               'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:68.0) Gecko/20100101 Firefox/68.0',
                               'Content-Type': 'application/json;charset=UTF-8'},
                      data=json.dumps(json_data))

    print(r.status_code)
    row_data = r.json()
    apt = Apartment()

    for i, deal in enumerate(row_data[spesific]):
        date_time_obj = datetime.datetime.strptime(deal['DEALDATE'], '%d.%m.%Y')

        apt.add_apt(size=deal['DEALNATURE'], price=deal['DEALAMOUNT'], floor=deal['FLOORNO'], year=date_time_obj.date(),
                    rooms=deal['ASSETROOMNUM'])

        print(apt.apartments[i])

    # plot price vs time list1 contains all elements of list2 all(elem in list1  for elem in list2)
    y = np.asarray([apt.apartments[i]['meter_price'] if apt.apartments[i]['size'] > 1 else 0 for i, _ in enumerate(apt.apartments)])
    dates = matplotlib.dates.date2num([apt.apartments[i]['year'] for i, _ in enumerate(apt.apartments)])
    matplotlib.pyplot.plot_date(dates, y)

    # Lin reg
    reg = LinearRegression().fit(dates.reshape(-1, 1), y.reshape(-1, 1))
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = reg.intercept_ + (reg.coef_ * x_vals)
    plt.plot(x_vals.reshape(-1, 1), y_vals.T, '--')

    # DeepRed
    regression = deep_reg(dates, y, figure=0, epochs=20000, adaptive=False, plot=False)

    plt.savefig('price_to time.png')


class Apartment:
    def __init__(self):
        self.apartments = []

    def add_apt(self, size=0, price=0, floor=0, year=None, rooms=0):
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        'en_US.UTF-8'
        self.apartments.append({"size": float(size) if size != '' else 0.1, "price": locale.atof(price),
                                "floor": floor, "year": year, "rooms": rooms})
        if self.apartments[-1]['size'] != 0:
            self.meter_price(-1)

    def meter_price(self, i):
        self.apartments[i]['meter_price'] = float(self.apartments[i]['price'] / self.apartments[i]['size'])


if __name__ == '__main__':
    main()
