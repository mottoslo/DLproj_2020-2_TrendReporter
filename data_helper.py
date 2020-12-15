# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from functools import reduce

# 이동평균, portion click 자동화 코드. optional
class DataProcessing():
    def __init__(self): pass

class InputData():
    def __init__(self, start_date = '2017-01-01', end_date = '2020-12-31'):
        '''
        [카테고리별 columns]
        0. clicks : 카테고리 cross scaling (0~1000)
        1-1. clicks_scaler : 카테고리 local scaling (0~1000)
        1-2. clicks_ratio : 첫번째 데이터 대비 변화율
        1-3. clicks_proportional : 이동평균 대비 변화율
        1-4. temp_avg : temp/100
        1-5. tmp_low : temp/100
        1-6. tpm_high : temp/100
        1-7. pollution : pollution/100
        1-8. sellings_lv1 : sellings_lv1/100
        1-9. sellings_lv2 : selling_lv2/100
        '''
        # 클릭수, 판매액지수 경상지수, 판매액지수 계절지수, 미세먼지, 기온 불러오기
        self.click_index = pd.read_csv("./data/naver_processed/NAVER_카테고리별_트렌드지수_20201202_minmax.csv", parse_dates=['date'], index_col='date', encoding='UTF8')
        self.sellings_defla = pd.read_csv("./data/sellings/판매액지수_20164R_20203R_경상지수.csv", parse_dates=['date'], index_col='date', encoding='cp949')
        self.sellings_season = pd.read_csv("./data/sellings/판매액지수_20164R_20203R_계절지수.csv", parse_dates=['date'], index_col='date', encoding='cp949')
        self.pollution = pd.read_csv("./data/weather/airpollution_170801_201113.csv", parse_dates=['date'], index_col='date',encoding='UTF8')
        self.temperature = pd.read_csv("./data/weather/temperature_170801_201113.csv", parse_dates=['date'], index_col='date',encoding='UTF8')

        # sellings 데이터 선형 보간
        self.sellings_defla.interpolate(inplace=True)
        self.sellings_season.interpolate(inplace=True)

        # 2016년도 drop
        # mask = self.sellings_defla.index >='2017-01-01'
        # self.sellings_defla = self.sellings_defla[mask]
        # self.sellings_season = self.sellings_season[mask]

        # 판매액지수 경상지수, 판매액지수 계절지수, 미세먼지, 기온 100으로 나누기
        self.sellings_defla /= 100
        self.sellings_season /= 100
        self.pollution /= 100
        self.temperature /= 100
        
        # param 기준으로 카테고리 이름과 통계청 분류 확인
        # _lookup_sellings = pd.read_excel("./data/naver_source/input_네이버 카테고리_filtered.xlsx", sheet_name='input', index_col='params', usecols=['params', 'kosis_lv1', 'kosis_lv2'])
        # _lookup_catgs = pd.read_csv("./data/naver_source/input_네이버 카테고리_filtered.csv", index_col='params', encoding='UTF8')
        # self.lookup =_lookup_sellings.join(_lookup_catgs, how='left', on='params')
        self.lookup = pd.read_excel("./data/naver_source/input_네이버 카테고리_filtered.xlsx", sheet_name='input', index_col='params')


    def processNsave(self):
        # 카테고리 단위로 click 테이블 처리
        catgs = self.click_index.columns
        for catg in catgs:
            param = self.lookup[self.lookup['catg_nm_full']==catg].index.values[0]
            kosis_catg_lv1 = self.lookup.loc[param, 'kosis_lv1']
            kosis_catg_lv2 = self.lookup.loc[param, 'kosis_lv2']

            df = pd.DataFrame()
            df['clicks'] = self.click_index.loc[:, catg]
            #1. minmax
            scaler =MinMaxScaler(feature_range = (0,100))
            df['clicks_minmax'] = scaler.fit_transform(df[['clicks']])
            #2. click/anchor ratio
            anchor = df['clicks'][0]
            df['clicks_first_ratio'] = (df['clicks'] - anchor) / anchor
            #3. ma ratio
            df['clicks_ma100'] = df['clicks'].rolling(window=100).mean()
            df.dropna(inplace=True)
            df['clicks_ma_ratio'] = (df['clicks'] - df['clicks_ma100'])/df['clicks_ma100']
            #drop clicks_ma100
            df.drop(columns = ['clicks_ma100'], inplace=True)

            # lookup sellings
            ss = self.sellings_season.loc[:, [kosis_catg_lv1, kosis_catg_lv2]]
            ss.rename(columns ={ss.columns[0] : 'kosis_catg_lv1_season', ss.columns[1] : 'kosis_catg_lv2_season'}, inplace = True)
            sd = self.sellings_defla.loc[:, [kosis_catg_lv1, kosis_catg_lv2]]
            sd.rename(columns={sd.columns[0]: 'kosis_catg_lv1_defla', sd.columns[1]: 'kosis_catg_lv2_defla'}, inplace=True)

            # merge
            tomerge = [df, ss, sd, self.temperature, self.pollution]
            df = reduce(lambda left, right: pd.merge(left, right, on=['date'], how='left'), tomerge)
            catg = catg.replace('/', '+')
            # save as csv
            df.to_csv(f"./data/catgwise/{param}_{catg}.csv", float_format = '%.4f')

temp = InputData()
temp.processNsave()