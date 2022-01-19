import json
from pathlib import Path
import os
import logging
import pandas as pd

# %% Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', handlers=[logging.StreamHandler()])

logging.info("Starting logging")


# %% Paths
path_input = Path(os.environ.get("INPUTS", "/data/inputs"))
path_output = Path(os.environ.get("OUTPUTS", "/data/outputs"))
path_logs = Path(os.environ.get("LOGS", "/data/logs"))
dids = json.loads(os.environ.get("DIDS", '[]'))
assert dids, f'no DIDS are defined, cannot continue with the algorithm'

did = dids[0]
input_files_path = Path(os.path.join(path_input, did))
input_files = list(input_files_path.iterdir())
first_input = input_files.pop()
# assert len(input_files) == 1, "Currently, only 1 input file is supported."
path_input_file = first_input
logging.debug(f'got input file: {path_input_file}, {did}, {input_files}')
path_output_file = path_output / 'predicted_values2022.csv'

# %% Check all paths
assert path_input_file.exists(), "Can't find required mounted path: {}".format(path_input_file)
assert path_input_file.is_file() | path_input_file.is_symlink(), "{} must be a file.".format(path_input_file)
assert path_output.exists(), "Can't find required mounted path: {}".format(path_output)
# assert path_logs.exists(), "Can't find required mounted path: {}".format(path_output)
logging.debug(f"Selected input file: {path_input_file} {os.path.getsize(path_input_file)/1000/1000} MB")
logging.debug(f"Target output folder: {path_output}")


# %% Load data
logging.debug("Loading {}".format(path_input_file))

with open(path_input_file, 'rb') as fh:
    df = pd.read_csv(fh)

logging.debug("Loaded {} records into DataFrame".format(len(df)))


#Remove null values
df2 = df.dropna()

#Remove outliers
min_price, max_price = df2.price.quantile([0.01, 0.99])
pop_cars = df2[(df2.price<max_price) & (df2.price>min_price)]


min_price, max_price = df2.mileage.quantile([0.01, 0.99])
pop_cars = pop_cars[(pop_cars.mileage<max_price) & (pop_cars.mileage>min_price)]


min_price, max_price = df2.hp.quantile([0.01, 0.999])
pop_cars = pop_cars[(pop_cars.hp<max_price) & (pop_cars.hp>min_price)]

#Create age variable to replace year
pop_cars['age'] = 2022 - pop_cars['year']

pop_cars.drop('year', axis=1, inplace=True)

#Adding mileage by 13500 according to research for assumption
pop_cars['mileage'] = pop_cars['mileage'] + 13500

#Create dummy variables for model and offertype to transform them into numerical values for model prediction
df3 = pd.get_dummies(pop_cars, columns=['make','offerType'])

#Predict price for 2022 using linear regression equation
pop_cars['Predicted Price 2022']=(-4.64537722e-02*df3['mileage']) + (1.15917656e+02*df3['hp']) + (-6.65847446e+03*df3['make_Abarth']) + (-1.21895474e+03*df3['make_Alfa'])+(7.61246867e+03*df3['make_Alpina']) + (2.53157311e+04*df3['make_Alpine']) + (2.62047040e+04*df3['make_Aston']) +(1.54578205e+03*df3['make_Audi'])+(-1.51539744e+03*df3['make_BMW']) + (-4.66864327e+03*df3['make_Baic']) + (-1.60922672e+03*df3['make_Cadillac']) + (-5.09501473e+03*df3['make_Chevrolet'])+(-9.45989936e+03*df3['make_Chrysler']) +(-3.97880674e+03*df3['make_Citroen']) +(-3.25837706e+03*df3['make_Cupra']) +(-5.65491842e+03*df3['make_DFSK'])+( 3.07711222e+03*df3['make_DS']) +(-5.25794728e+03*df3['make_Dacia']) +(-1.50210262e+03*df3['make_Daihatsu']) +(-7.88047667e+03*df3['make_Dodge'])+(2.16842855e+04*df3['make_FISKER']) +(-3.87297724e+03*df3['make_Fiat']) +(-3.88456152e+03*df3['make_Ford']) +(-2.72485540e+03*df3['make_Honda'])+(-4.04664496e+03*df3['make_Hyundai']) +(-5.95579580e+03*df3['make_Infiniti']) +(-1.78022316e+03*df3['make_Isuzu']) +(-1.09993075e+03*df3['make_Iveco'])+(1.58321731e+03*df3['make_Jaguar']) +(-3.62062293e+02*df3['make_Jeep']) +(-3.88490924e+03*df3['make_Kia']) +(-4.78673159e+03*df3['make_Lada'])+(-4.32282642e+03*df3['make_Lancia']) +(1.21333095e+04*df3['make_Land']) +(-4.56811820e+02*df3['make_Lexus']) +(-2.30322081e+03*df3['make_MINI'])+(3.05156354e+03*df3['make_Maserati']) +(-3.60520395e+03 *df3['make_Mazda']) +( 1.87809623e+03*df3['make_Mercedes-Benz']) +(-5.36806212e+03*df3['make_Mitsubishi'])+(3.59452994e+04*df3['make_Morgan']) +(  -3.50749999e+03*df3['make_Nissan']) +(-4.64278252e+03*df3['make_Opel']) +(-3.83323124e+03*df3['make_Peugeot'])+(-2.27373675e-11*df3['make_Piaggio']) +(5.90320616e+03*df3['make_Polestar']) +( 1.32329398e+04*df3['make_Porsche']) +(-1.14152290e+04*df3['make_RAM'])+(-4.56962289e+03*df3['make_Renault']) +(-3.22069986e+03*df3['make_SEAT']) +( -2.61559985e+03*df3['make_Skoda']) +(-6.91354431e+03*df3['make_SsangYong'])+(-1.79020308e+03*df3['make_Subaru']) +(-4.69401794e+03*df3['make_Suzuki']) +( -1.46246096e+03*df3['make_Tesla']) +(-3.30199420e+03*df3['make_Toyota'])+(-1.93284395e+02*df3['make_Volkswagen'])+(2.43030320e+03*df3['make_Volvo'])+(-3.22479192e+03*df3['make_smart'])+(3.90808492e+03*df3['offerType_Demonstration']) +(1.19549322e+03*df3["offerType_Employee's car"]) +(-3.53753756e+03*df3['offerType_New']) +(-8.29042685e+02*df3['offerType_Pre-registered'])+( -7.36997905e+02*df3['offerType_Used'])+(-1.11734228e+03*df3['age']) + 13180.134917603684

#Create new variable to show price difference between 2021 and 2022
pop_cars['Price Difference'] = pop_cars['Predicted Price 2022'] - pop_cars['price']

#Renaming columns
pop_cars = pop_cars.rename(columns={"mileage": "Mileage", "make": "Make", "model": "Model", "fuel": "Fuel", "gear": "Gear", "offerType": "OfferType", "price": "Original Price 2021", "hp": "Hp", "age": "Age"})


logging.debug("Built summary of records.")

dfvo.to_csv(path_output_file)

logging.debug("Wrote results to {}".format(path_output_file))

logging.debug("FINISHED ALGORITHM EXECUTION")





