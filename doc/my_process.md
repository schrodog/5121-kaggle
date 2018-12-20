# Note
{ } = explanation

## Data Cleaning
### remove outlier
lotFrontage, LotArea, MasVnrArea
- where data range exist only in training, not testing data
combine test & train dataset

### find missing train data
LotFrontage: plot median, mean; group by neighborhood
Alley: 'NA' ??
MasVnrType: most frequent
Bsmt*: string->'NA'
BsmtFinSF1/2: 0
BsmtFinType1, BsmtFinType2: 'Unf'
  {basement of type1 and type2 are different, one can already finished and another is not finished}

Electrical: missing 1, {almost normal data} => most freq
FireplaceQu: 'NA'
  { refer to no fireplace, Fireplaces=0}
PoolQC: 'NA'
  {becoz poolArea=0 for all NA}
MiscFeature,Fence: 'NA'
GarageFinish, GarageType, GarageQual, GarageType: 'NA'

### fill test data
1459 records
- missing values in train and test data are different
      
MSZoning        1455 [majority]
Exterior1st     1458 [group by RoofMatl]
Exterior2nd     1458 [group by RoofMatl]
BsmtFinSF1      1458 [majority]
BsmtFinSF2      1458 [majority]
BsmtUnfSF       1458 [majority]
TotalBsmtSF     1458 [majority]
BsmtFullBath    1457 [group by FullBath, HalfBath]
BsmtHalfBath    1457 [group by FullBath, HalfBath]
KitchenQual     1458 [group by KitchenAbvGr]
Functional      1457 [majority]
GarageCars      1458 [group by GarageType]
GarageArea      1458 [group by GarageType, take median]
SaleType        1458 [majority]


## Feature engineering
### add new features
1. year 
RemodAge = YrSold - YearRemodAdd
HouseAge = YrSold - YearBuilt
Oldness = HouseAge*0.5 + RemodAge
GarageAge = -1 Or YrSold - max(GarageYrBlt, YearRemodAdd)

2. grade, value
OverallValue = OverallQual + OverallCond
GarageGrade = GarageQual + GarageCond
ExterValue = ExterQual + ExterCond
KitchenValue = KitchenAbvGr * KitchenQual
FireplaceValue = Fireplaces * FireplaceQu
GarageValue = GarageArea * GarageQual * GarageFinish

BsmtValue = BsmtFinType1*BsmtFinSF1 + BsmtFinType2*BsmtFinSF2 + 0.2*BsmtUnfSF + BsmtCond*BsmtQual*TotalBsmtSF*0.3
{BsmtFinSF2: >0 count=167, BsmtFinSF1 >0 count=467
BsmtUnfSF >0 count=1342
}

BathValue = BsmtFullBath + 0.5*BsmtHalfBath + 2*FullBath + 1.5*HalfBath

LotValue = LotArea*LotFrontage 

3. total area
TotalPorchSF = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch
{count_0: ScreenPorch=1344, 3SsnPorch=1436}

TotalSF = 1stFlrSF + 2ndFlrSF + GrLivArea + 0.4*(LowQualFinSF + TotalBsmtSF) + 0.1*(WoodDeckSF + TotalPorchSF + PoolArea + LotArea + GarageArea) + LotArea


### change features
1. find features with very low var
- change to binary variable
'RoofStyle', 'LotShape', 'MSZoning', 'Electrical', 'Condition2', 'RoofMatl', 'Alley', 'Heating', 'LandSlope', 'MiscFeature', 'CentralAir', 'PoolQC', 'Street', 'Utilities'

RoofStyle -> Gable, Hip, others
LotShape -> Reg, IR1, others
LandSlope -> Gtl, others
MSZoning -> RL, RM, FV, others
Heating -> GasA, others
Alley -> Yes,No
PoolExist -> Yes,No
  {PoolArea only 7 record, 400 < x < 750 }
Electrical -> SBrkr, others

### Drop features
CentralAir
PoolQC
Condition2
RoofMatl
Street
Utilities
MiscFeature {Only Shed is meaningful}

### Discretize (num -> class)
MiscVal: 0 -> 0, >0 -> 1
 {1408 records == 0}

OverallQual -> 1-3,4-7,8-10
OverallCond -> 0-5
GarageQual, GarageCond -> 0-5
ExterQual, ExterCond -> 0-5
KitchenAbvQual, KitchenCond -> 0-5
FireplaceQu -> 0-5
Neighborhood -> 0-4   {use KMeans to group, 5 is best}


### OneHot Encoding
Integer Encoding: for field with natural ordering, comparable
OneHot Encoding: no apparent ordering
- discover month more house sold, price inverse with sold count
- price,count ~ MSSubClass

## Scale features
SalePrice -> log(1+x)
all other features already become numbers => RobustScale


## outlier
### exist in train, not test

TotalSF > 100,000 [4]

LotFrontage > 200
LotArea > 100,000

MasVnrArea > 1500

1. drop rows missing in test data
exterior1st (imstucc,stone), ...

## important features
### after restriction by XGBoost 
OverallQual
GrLivArea
TotalSF
GarageValue
OverallValue
BsmtValue
HouseAge
1stFlrSF
TotalBsmtS
Oldness
LotFrontage
TotalPorchSF
GarageAre
FireplaceValue
LotArea
GarageYrBlt
BathValu
ExterQual
2ndFlrSF
KitchenQual


### by XGBoost
LotFrontage
GrLivArea
TotalSF
BsmtValue
OverallCond
LotArea
Oldness
OverallValue
1stFlrSF
OverallQual

TotalBsmtSF
Functional
GarageArea
YearBuilt
BsmtFinSF1
GarageYrBlt
YearRemodAdd
GarageValue
2ndFlrSF
HouseAge

RemodAge
OpenPorchSF
BsmtExposure
BsmtUnfSF
TotalPorchSF
-----
diff:

LotArea
WoodDeskSF
MonthSaledCount
MonthSaledMeanPrice
MasVnrArea
LotArea-S2

### by MINE
use MIC, (maximum information coefficient)

['OverallQual', 0.5607558959551385],
['ExterQual', 0.4894627978060431],
['GarageValue', 0.4891427673261014],
['GrLivArea', 0.4845168975576535],
['GarageAge', 0.4614426655064794],
['KitchenQual', 0.45472221034923543],
['FullBath', 0.43809982011943255],
['BsmtQual', 0.43542394796184447],
['BathValue', 0.43202497986898086],
['HouseAge', 0.4126891991394268],
['YearBuilt', 0.4080867598152763],
['GarageCars', 0.40695379629506917],
['Oldness', 0.4050056942712898],
['GarageFinish', 0.4011832931810155],
['GarageArea', 0.3967203817544145],
['KitchenValue', 0.382648832485799],
['ExterValue', 0.36583929759853245],
['FireplaceValue', 0.35894074606131937],
['FireplaceQu', 0.3566699060952901],
['Foundation_PConc', 0.35390809708048115],
['OverallValue', 0.34619210141355805],
['GarageYrBlt', 0.3407989743834871],
['TotalBsmtSF', 0.3401545667869029],
['BsmtValue', 0.338028409514795],
['Fireplaces', 0.3281733570536026],
['TotalSF', 0.31562315718579015],
['1stFlrSF', 0.3004397810082837],
['RemodAge', 0.2926741146528306],
['YearRemodAdd', 0.28969024646250424],
['OpenPorchSF', 0.2817148580971405],
['Neighborhood_2', 0.2742739031272732],
['HeatingQC', 0.27262361164053184],
['GarageType_Attchd', 0.2709341502065679],
['TotRmsAbvGrd', 0.26665366329697016],
['MSSubClass_60', 0.262915937157715],
['LotArea', 0.25723374369047064],
['2ndFlrSF', 0.24676351277251868],
['GarageType_Detchd', 0.24501025415538208],
['Foundation_CBlock', 0.23321418915247152],
['Exterior2nd_VinylSd', 0.21218151402371083],
['MasVnrType_None', 0.20810188769043536],
['Exterior1st_VinylSd', 0.20717773282317156],
['GarageGrade', 0.2066014557717289],
['MasVnrArea', 0.20471261221835307],
['MSZoning_1', 0.20237957821988165],
['Neighborhood_0', 0.2000896451912745],
['LotFrontage', 0.1999896666622148],
['HalfBath', 0.19850549414199556],
['BsmtFinSF1', 0.19351804512526782]]

- basically 2 groups overlap, so choose top 50




