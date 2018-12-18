# process
remove Id, SalePrice

- remove outlier
lotFrontage, LotArea, MasVnrArea
- where data range exist only in training, not testing data

combine test & train dataset

- find missing data, fill missing
LotFrontage: plot median, mean; group by neighborhood

Alley: 'NA'
MasVnrType: most frequent
Bsmt*: string->'NA'
BsmtFinSF1/2: 0
Electrical: missing 1, most freq
FireplaceQu,PoolQC,MiscFeature,Fence: 'NA'
Garage*: 'NA'
MSZoning
Utilities
Exterior1st/2nd
KitchenQual
Functional
SaleType
Functional

Feature engineering:
YearBuilt - GarageYrBuilt => LinearRegression
fill GarageYrBlt 'NA' with prediction

RemodYear = YearRemodAdd - YearBuilt
HasRemodeled = YearRemodAdd != YearBuilt (remodeling happen from built)
HasRecentRemodel = YearRemodAdd == YrSold (remodeling happen when house sold)

GarageBltYears = GarageYrBlt - YearBuilt
Now_YearBuilt = 2017 - YearBuilt
Now_YearRemodAdd = 2017 - YearRemodAdd
Now_GarageYrBlt = 2017 - GarageYrBlt

- discover month more house sold, price inverse with sold count
- price,count ~ MSSubClass

- Encode comparable label with num
Street, Alley, ...

Neighborhood location -> numerical latitude, longitude -> NeighborDistance
NeighborDistance = sqrt( (longitude-min)^2 + (latitude-min)^2 )

bin Neighborhood [0-4]
NeighborPrice, NeighborBin

- separate string,number features
create binary fields on popular features
eg. LotShape, LandContour ...

- simplify existing feature, discretize
OverallQual: 1-3, 4-6, 7-10
OverallCond: 1-3, 4-6, 7-10

OverallGrad = OverallQual * OverallCond
GarageGrade = GarageQual * GarageCond
ExterGrade = ExterQual*ExterCond
KitchenScore = KitchenAbvGr*KitchenQual
FireplaceScore = Fireplaces*FireplaceQu
GarageScore = GarageArea*GarageQual
PoolScore = PoolArea*PoolQC

TotalBath = BsmtFullBath + 0.5*BsmtHalfBath + FullBath + 0.5*HalfBath
TotalPorchSF = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch
AllSF = GrLivArea + TotalBsmtSF + TotalPorchSF + WoodDeckSF + PoolArea

BoughtOffPlan = SaleCondition(abnormal,alloca,..) -> 0

- find important feature by XGBRegressor
15 most import feature
d^2, d^3, sqrt(d)

- for boolean features, do not scatter and skewed
use max scater, 95% quantile to scale data

- transform skewed numeric features by taking log(feature+1)
- make features more normal
skewed > 0.75
NeighborPrice, NeighborPrice-s2, NeighborPrice-s3 -> log(1+x)
MonthSaledMeanPrice, MSSubClassMeanPrice, NeighborPrice.. -> log(1+x)

- drop columns missing in test data
exterior1st_imstucc,stone, ...





