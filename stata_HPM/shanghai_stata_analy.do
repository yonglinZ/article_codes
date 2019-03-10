set more off
reg HP_LN center_d area year ori hs pr pf gr build airport_d bus_d subway_d train_d financia_d food_d hospital_d sciedu_d shop_d tour_d water_d water_area,r
vif
gen id=_n
tsset id
estat dwatson
